//
//  DetailsViewController.swift
//  Yale Spring Fling
//
//  Created by Alexis Dornan on 10/12/19.
//

import UIKit
import Firebase

class DetailsViewController: UIViewController {
    
    var lineupSlot : LineupSlot!
    @IBOutlet weak var stayTunedLabel: UILabel!
    
    @IBOutlet weak var checkOutLabel: UILabel!
    
    @IBOutlet weak var nameLabel: UILabel!
    
    @IBOutlet weak var timeLabel: UILabel!
    
    @IBOutlet weak var artistPic: UIImageView!
    
    
    @IBOutlet weak var bioLabel: UITextView!
    
    @IBOutlet weak var spotifyButton: UIButton!
    
    var spotifyLink : String!
    
    @objc func goToSpotify(){
        if let url = URL(string:spotifyLink) {
            UIApplication.shared.open(url)
        }
    
    
    }
    override func viewDidLoad() {
        super.viewDidLoad()
        nameLabel.text = lineupSlot.name
        timeLabel.text = lineupSlot.startTime + " - " + lineupSlot.endTime
        // Do any additional setup after loading the view.
        if let pic = lineupSlot.picture{
            downloadImage(image_ref: pic, imageView: artistPic)
        }
        if let spotify = lineupSlot.spotifyLink{
            spotifyLink = spotify
            spotifyButton!.addTarget(self, action: #selector(goToSpotify), for: .touchUpInside)
        }
        else{
            spotifyButton.isHidden = true
        }
        // bio is an optional so possible error here
        bioLabel.text = lineupSlot.bio
        bioLabel.isEditable = false
        
       
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
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
