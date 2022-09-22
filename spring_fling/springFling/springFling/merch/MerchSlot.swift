//
//  MerchSlot.swift
//  springFling
//
//  Created by Alexis Dornan on 11/27/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit

class MerchSlot{
    
    var itemName : String
    var picture : String?
    var description : String?
    var buyNowLink : String?
    var price : String?
    var colors : String?
    var sizes : String?
    var database_name : String?
        
    init(itemName: String, picture: String?, description : String?, buyNowLink: String?, price: String?, colors: String?, sizes: String?, database_name: String?){
            self.itemName = itemName
            self.picture = picture
            self.description = description
            self.buyNowLink = buyNowLink
            self.price = price
            self.colors = colors
            self.sizes = sizes
            self.database_name = database_name
    }
}

