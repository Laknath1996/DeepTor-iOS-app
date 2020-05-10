//
//  ViewController.swift
//  DeepTor
//
//  Created by Ashwin de Silva on 5/9/20.
//  Copyright © 2020 Ashwin de Silva. All rights reserved.
//

import UIKit
import Vision


class ViewController: UIViewController, UITextFieldDelegate, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    //MARK: UI Properties
    @IBOutlet weak var nameTextField: UITextField!
    @IBOutlet weak var subjectNameLabel: UILabel!
    @IBOutlet weak var photoImageView: UIImageView!
    @IBOutlet weak var subjectImageLabel: UILabel!
    
    //MARK : Core ML Model
    let SegmentationModel = ScleraUNet()
    
    //MARK : Vision Properties
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // set up the model
        print("Setting up model...")
        setupModel()
        print("Model is ready")
        
        // Handle the text field’s user input through delegate callbacks.
        nameTextField.delegate = self
    }

    //MARK: UITextFieldDelegate
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        // Hide the keybord
        textField.resignFirstResponder()
        return true
    }
    
    func textFieldDidBeginEditing(_ textField: UITextField) {
        let a = "Select "
        let b = "'s External Eye Image :"
        subjectImageLabel.text = a + nameTextField.text! + b
    }
    
    //MARK: UIImagePickerControllerDelegate
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        // Dismiss the picker if the user canceled.
        dismiss(animated: true, completion: nil)
    }
    
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
        // The info dictionary may contain multiple representations of the image. You want to use the original.
        guard let selectedImage = info[UIImagePickerController.InfoKey.originalImage] as?  // check
            UIImage else {
                fatalError("Expected a dictionary containing an image, but was provided the following: \(info)")
        }
        
        // Set photoImageView to display the selected image.
        photoImageView.image = selectedImage
        
        // Dismiss the picker
        dismiss(animated: true, completion: nil)
    }
    
    // MARK: Setup the Core ML model
    func setupModel(){
        if let visionModel =  try? VNCoreMLModel(for: SegmentationModel.model){
            print("Model all good")
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
    }
    
    //MARK: Actions
    @IBAction func selectImageFromPhotoLibrary(_ sender: UITapGestureRecognizer) {
        // Hide the keyboard
        nameTextField.resignFirstResponder()
        
        // UIImagePickerController is a view controller that lets a user pick media from their photo library.
        let imagePickerController = UIImagePickerController()
        
        // only allow photos to be picked, not taken.
        imagePickerController.sourceType = .photoLibrary
        
        // Make sure ViewController is notified when the user picks an image.
        imagePickerController.delegate = self
        
        present(imagePickerController, animated: true, completion: nil)
    }
    
    @IBAction func setSubjectName(_ sender: UIButton) {
        subjectImageLabel.text = "Select the External Eye Image :"
        nameTextField.text = ""
    }
    
    @IBAction func segementImage(_ sender: UIButton){
        print("Segmenting... ")
        let input_image = photoImageView.image?.cgImage
        predict(with: input_image!)
    }
    
}

//MARK: Inference
extension ViewController {
    
    func predict(with cgimage : CGImage){
        print("Predicting...")
        guard let request = request else { fatalError() }
        let handler = VNImageRequestHandler(cgImage: cgimage, options: [:])
        try? handler.perform([request])
        print("Prediction done")
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?){
        if let observations = request.results as? [VNCoreMLFeatureValueObservation], let
            segmentationMap = observations.first?.featureValue.multiArrayValue{
        
            
            for i in 0...255 {
                for j in 0...255 {
                    let offset = 0 * segmentationMap.strides[0].intValue +
                                 i * segmentationMap.strides[1].intValue +
                                 j * segmentationMap.strides[2].intValue
                    if segmentationMap[offset].doubleValue >= 0.5 {
                        segmentationMap[offset] = 1
                    } else {
                        segmentationMap[offset] = 0
                    }
                }
            }
            
            photoImageView.image = segmentationMap.image(min: 0, max: 1)
            print("Done!")
        }
    }
}
