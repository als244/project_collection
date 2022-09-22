//
//  Clothing.swift
//  springFling
//
//  Created by Alexis Dornan on 11/27/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit

class Clothing{
    
    var itemName : String
    var picture : UIImage?
    var description : String?
    var buyNowLink : String?
    var price : String?
    var colors : String?
    var sizes : String?
    
    init(itemName: String, picture: UIImage?, description : String?, buyNowLink: String?, price: String?, colors: String?, sizes: String?){
        self.itemName = itemName
        self.picture = picture
        self.description = description
        self.buyNowLink = buyNowLink
        self.price = price
        self.colors = colors
        self.sizes = sizes
    }
}
