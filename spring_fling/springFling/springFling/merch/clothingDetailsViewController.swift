//
//  clothingDetailsViewController.swift
//  springFling
//
//  Created by Alexis Dornan on 11/27/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import Firebase
import UIKit

class clothingDetailsViewController: UIViewController {
    
    var merchSlot : MerchSlot!
//    @IBOutlet weak var stayTunedLabel: UILabel!

    @IBOutlet weak var priceLabel: UILabel!

    @IBOutlet weak var colorsLabel: UILabel!

    @IBOutlet weak var nameLabel: UILabel!
    
    @IBOutlet weak var sizesLabel: UILabel!
    
    @IBOutlet weak var clothingPic: UIImageView!
    
    @IBOutlet weak var descriptionLabel: UILabel!
    
    @IBOutlet weak var buyNowButton: UIButton!
    
    var buyNowLink : String!
    
    @objc func goToWebsite(){
        if let url = URL(string:buyNowLink) {
            UIApplication.shared.open(url)
        }
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        nameLabel.text = merchSlot.itemName
        // Do any additional setup after loading the view.
        if let pic = merchSlot.picture{
            downloadImage(image_ref: pic, imageView: clothingPic)
        }
        if let website = merchSlot.buyNowLink{
            buyNowLink = website
            buyNowButton!.addTarget(self, action: #selector(goToWebsite), for: .touchUpInside)
//            stayTunedLabel.isHidden = true
        }
        else{
            buyNowButton.isHidden = true
//            checkOutLabel.isHidden = true
//            stayTunedLabel.isHidden = false
        }
        // bio is an optional so possible error here
        descriptionLabel.text = merchSlot.description
        priceLabel.text = merchSlot.price
        colorsLabel.text = merchSlot.colors
        sizesLabel.text = merchSlot.sizes
    }
    
    
    func downloadImage(image_ref: String?, imageView: UIImageView){
           
           if let img_ref = image_ref {
               let storageRef = Storage.storage()
                                                                  
               let storage = storageRef.reference(forURL: "gs://spring-fling-39c0c.appspot.com")
                                           
               // Create a reference to the file you want to download
               let storageImageRef = storage.child(img_ref)
               
               // Download in memory with a maximum allowed size of 1MB (1 * 1024 * 1024 bytes)
               storageImageRef.getData(maxSize: 5 * 1024 * 1024) { data, error in
               if let error = error {
                   print(error)
                   } else {
                   // Data for image is returned
                   imageView.image = UIImage(data: data!)
                   }
               }
           }
       }
}
