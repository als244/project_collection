//
//  Announcement.swift
//  YaleSpringFling
//
//  Created by Andrew Sheinberg on 10/15/19.
//

import Foundation
import UIKit

class Announcement{
    
    var message: String
    var database_name: String
    var datetime: Date
    var author: String?
    var picture: UIImage?
    
    
    init(database_name: String, message: String, datetime: Date, author: String?, picture: UIImage?) {
        self.database_name = database_name
        self.message = message
        self.datetime = datetime
        self.author = author
        self.picture = picture
    }
    
}
