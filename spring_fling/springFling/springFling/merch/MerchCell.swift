//
//  MerchCell.swift
//  springFling
//
//  Created by Alexis Dornan on 11/27/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit
import Firebase

class MerchCell: UITableViewCell {
    
    @IBOutlet weak var clothingPic: UIImageView!
    @IBOutlet weak var clothingNameLabel: UILabel!
    
    func setMerchSlot(slot : MerchSlot){
        if let pic = slot.picture{
            downloadImage(image_ref: pic, imageView: clothingPic)
        }
        clothingNameLabel.text = slot.itemName
    }
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
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
