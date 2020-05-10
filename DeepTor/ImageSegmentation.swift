//
//  ImageSegmentation.swift
//  DeepTor
//
//  Created by Ashwin de Silva on 5/9/20.
//  Copyright © 2020 Ashwin de Silva. All rights reserved.
//

import Foundation
import Vision
import CoreML
import UIKit

class ImageSegmentation: UIViewController {
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    var cgImage: CGImage?
    
    let segmentationModel = ScleraUNet()
    
    let inputImage: UIImageView = {
        let image = UIImageView(image: UIImage())
        image.translatesAutoresizingMaskIntoConstraints = false
        image.contentMode = .scaleAspectFit
        return image
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUpModel()
        self.predict(with: self.cgImage!)
    }
    
//    override func viewDidAppear(_ animated: Bool) {
//        super.viewDidAppear(animated)
//        setUpModel()
//        self.predict(with: self.cgImage!)
//    }
    
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: segmentationModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .centerCrop
        } else {
            fatalError()
        }
    }
}
    
extension ImageSegmentation {
    
    func predict(with cgImage: CGImage) {
        guard let request = request else { fatalError() }
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let segmentationMap = observations.first?.featureValue.multiArrayValue {
        }
    }
}
