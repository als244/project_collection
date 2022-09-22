//
//  Artist.swift
//  Yale Spring Fling
//
//  Created by Alexis Dornan on 10/10/19.
//

import UIKit

class Artist{
    
    var firstName : String
    var lastName : String
    var picture : UIImage?
    
    var bio : String?
    
    var spotifyLink : String?
    
    init(firstName: String, lastName: String, picture: UIImage?, bio : String?, spotifyLink: String?){
        self.firstName = firstName
        self.lastName = lastName
        self.picture = picture
        self.bio = bio
        self.spotifyLink = spotifyLink
    }
}
