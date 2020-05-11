//
//  ViewController.swift
//  DeepTor
//
//  Created by Ashwin de Silva on 5/9/20.
//  Copyright © 2020 Ashwin de Silva. All rights reserved.
//

import UIKit
import Vision
import Foundation


class ViewController: UIViewController, UITextFieldDelegate, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    //MARK: UI Properties
    @IBOutlet weak var nameTextField: UITextField!
    @IBOutlet weak var subjectNameLabel: UILabel!
    @IBOutlet weak var photoImageView: UIImageView!
    @IBOutlet weak var subjectImageLabel: UILabel!
    
    //MARK : Core ML Model
    let SegmentationModel = ScleraUNnet()
    
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
            request?.imageCropAndScaleOption = .scaleFill
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
        let image = resizeImage(image: photoImageView.image!, targetSize: CGSize(width: CGFloat(256.0), height: CGFloat(256.0)))
        let input_image = image.cgImage
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
    
    func resizeImage(image: UIImage, targetSize: CGSize) -> UIImage {
        let size = image.size
        
        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        
        // Figure out what our orientation is, and use that to form the rectangle
        var newSize: CGSize
        if(widthRatio > heightRatio) {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }
        
        // This is the rect that we've calculated out and this is what is actually used below
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        
        // Actually do the resizing to the rect using the ImageContext stuff
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        image.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?){
        if let observations = request.results as? [VNCoreMLFeatureValueObservation], let
            segmentationMap = observations.first?.featureValue.multiArrayValue{
        
            print(segmentationMap.shape)
            
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
