//
//  EditAnnouncementViewConroller.swift
//  springFling
//
//  Created by Andrew Sheinberg on 11/4/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import Foundation
import UIKit
import Firebase


class ContentAnnouncementViewController: UIViewController {
    
    var announcement: Announcement!
    
    
    @IBOutlet weak var content: UITextView!
    
    
    @IBOutlet weak var typeBelowLabel: UILabel!
    
    @IBAction func submit(_ sender: Any) {
        
        var ref: DatabaseReference!
        ref = Database.database().reference()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "M/d/yyyy h:mm a"
        
        if announcement != nil {
            ref.child("announcements").child("slots").child(announcement.database_name).setValue(["content": content.text as String, "timestamp":dateFormatter.string(from: announcement.datetime)])
            
        } else{
            
            
            guard let key = ref.child("announcements").child("slots").childByAutoId().key else { return }
            
           
            
            let timestamp = dateFormatter.string(from: Date())
            ref.child("announcements").child("slots").child(key).setValue([ "content":content.text as String, "timestamp":timestamp])
            
        }
        

        self.dismiss(animated: true, completion: nil)
        
        
    }
    
    override func viewDidLoad() {
        self.hideKeyboardWhenTappedAround()
        content.textContainerInset = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        if announcement != nil {
            content.text = announcement.message
            typeBelowLabel.text = "Edit the Announcement Below:"
        }
        else{
            typeBelowLabel.text = "Add an Announcement Below:"
           
        }
        
        
    }
}
