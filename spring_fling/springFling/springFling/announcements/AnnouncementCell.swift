//
//  AnnouncementCell.swift
//  YaleSpringFling
//
//  Created by Andrew Sheinberg on 10/15/19.
//

import UIKit
import Foundation

class AnnouncementCell: UITableViewCell {

    
    
    @IBOutlet weak var message: UILabel!
    @IBOutlet weak var datetimeLabel: UILabel!
    
    func setAnnouncment(announcement: Announcement){
        
        message.text = announcement.message
        let formatter = DateFormatter()
        formatter.dateFormat = "M/d/yyyy h:mm a"
        datetimeLabel.text = formatter.string(from: announcement.datetime)
    }
    
    
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
