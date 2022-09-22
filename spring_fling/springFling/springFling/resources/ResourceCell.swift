//
//  ResourceCell.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 9/27/19.
//  Copyright © 2019 Project T. All rights reserved.
//

import UIKit

class ResourceCell: UITableViewCell {

    
    @IBOutlet weak var label: UILabel!
    

    func setResource (resource: Resource){
        label.text = resource.name
        
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



/*
 // MARK: - Navigation
 
 // In a storyboard-based application, you will often want to do a little preparation before navigation
 override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
 // Get the new view controller using segue.destination.
 // Pass the selected object to the new view controller.
 }
 */

