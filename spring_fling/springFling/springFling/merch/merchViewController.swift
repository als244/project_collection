//
//  merchViewController.swift
//  springFling
//
//  Created by Alexis Dornan on 9/7/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit
import Firebase
import FirebaseStorage

class merchViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    
    @IBOutlet weak var tableView: UITableView!
    
    private var showroomItems = [MerchSlot]()

    override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "Shop"
        populateShowroom()
        
        //populateShowroom()
        tableView.dataSource = self
        tableView.delegate = self
        // Do any additional setup after loading the view.
    }
    
//    override func viewWillAppear(_ animated: Bool) {
//
//    }
    
    func populateShowroom() {
        var ref: DatabaseReference!
        ref = Database.database().reference()
        ref.child("merch").child("slots").observeSingleEvent(of: .value, with: { (snapshot) in
            // Get all questions and answers
            var showroomItems : [MerchSlot] = []
            
            let value = snapshot.value as? NSDictionary
            for (key, slot) in value! {
                
                let merch_data = slot as? NSDictionary
                
                
                let merch_slot = MerchSlot(itemName: merch_data?["item_name"] as! String, picture: merch_data?["picture"] as? String, description: merch_data?["description"] as? String, buyNowLink: merch_data?["buyNow_link"] as? String, price: merch_data?["price"] as? String, colors: merch_data?["colors"] as? String, sizes: merch_data?["sizes"] as? String, database_name: key as? String)
                
                showroomItems.append(merch_slot)
            }
               
            print(showroomItems)
            self.showroomItems = showroomItems
            self.tableView.reloadData()

        })
        
    }

   
//    func createShowroom() -> [MerchSlot]{
//
//        let goldengoose = Clothing(itemName: "Superstar Sneakers", picture: UIImage(named: "shoe"), description: "These white-and-gold Golden Goose sneakers are a little bit more toned down than the brand's usual, but still have the cool, broken-in style that has made them one of our forever favorites.", buyNowLink: "https://www.shopbop.com/star-sneakers-golden-goose/vp/v=1/1540851516.htm?gclid=CjwKCAiA_f3uBRAmEiwAzPuaM7aTd_WinZB5UVI-nUJ5xE1GSIGyTuFTfQ36LfHmAszKJxKPDd0n8BoCbOEQAvD_BwE&currencyCode=USD&extid=SE_froogle_SC_usa&cvosrc=cse.google.GOOSE20619&cvo_campaign=SB_Google_USD&ef_id=CjwKCAiA_f3uBRAmEiwAzPuaM7aTd_WinZB5UVI-nUJ5xE1GSIGyTuFTfQ36LfHmAszKJxKPDd0n8BoCbOEQAvD_BwE:G:s", price: "$100", colors: "Colors: red, white, and orange", sizes: "Sizes: Women's 6, 7, 8, 9")
//
//        let yale = Clothing(itemName: "Corduroy Yale Hat", picture: UIImage(named: "hat"), description: "Inspired by the fun and stylish fabrics of yesteryear, our corduroy ball cap is a true vintage throwback making a reappearance in a big way. The small wale cords are right on trend, available in our classics low profile, relaxed adjustable fit and available in four warm and stylish colors. Imported.", buyNowLink: "https://yale.bncollege.com/shop/ProductDisplay?topCatId=40000&level=&imageId=1846805&urlRequestType=Base&productId=400000470345&catalogId=10001&categoryId=40402&parentCatId=40017&graphicId=VE02434502&langId=-1&storeId=16556", price: "$27", colors: "Colors: black, blue, and white", sizes: "One Size")
//
//        let slot1 = MerchSlot(clothing: goldengoose)
//
//        let slot2 = MerchSlot(clothing: yale)
//
//        return [slot1, slot2]
//
//    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        guard let clothingDVC = segue.destination as? clothingDetailsViewController else {return}
        let merchSlot = showroomItems[tableView.indexPathForSelectedRow!.row]
        tableView.deselectRow(at: tableView.indexPathForSelectedRow!, animated: true)
        clothingDVC.merchSlot = merchSlot
    }

    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.showroomItems.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "MerchCell", for: indexPath) as! MerchCell
        
        cell.setMerchSlot(slot : self.showroomItems[indexPath.row])
        
        return cell
    }
}
