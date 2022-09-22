//
//  LineupCell.swift
//  Yale Spring Fling
//
//  Created by Alexis Dornan on 10/10/19.
//

import Firebase
import UIKit

class LineupCell: UITableViewCell {
    
    @IBOutlet weak var artistPic: UIImageView!
    
    @IBOutlet weak var timeLabel: UILabel!
    
    @IBOutlet weak var nameLabel: UILabel!
    
    func setLineupSlot(slot : LineupSlot){
        if let pic = slot.picture{
            downloadImage(image_ref: pic, imageView: artistPic)
        }
        
        nameLabel.text = slot.name
        
        timeLabel.text = slot.startTime + " - " + slot.endTime
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
