//
//  Resource.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 9/28/19.
//

import UIKit
class Resource {
    
    var name: String
    var phoneNumber: String?
    var link: String?
    var picture: UIImage?
    
    init(name: String, phoneNumber: String?, link: String?, picture : UIImage?){
        self.name = name
        self.phoneNumber = phoneNumber
        self.link = link
        self.picture = picture
    }
}


