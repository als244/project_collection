//
//  AnnouncementsViewController.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 10/15/19.
//

import UIKit
import Foundation
import Firebase

class AnnouncementsViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {

    @IBOutlet weak var tableView: UITableView!

    
    var announcements  = [Announcement]()
    var admin : Bool?
    let refreshControl = UIRefreshControl()
    
    override func viewDidLoad() {
        authenticate()
        super.viewDidLoad()
        self.navigationItem.setHidesBackButton(true, animated: false)
        refreshControl.addTarget(self, action: #selector(handleRefresh), for: .valueChanged)
        tableView.addSubview(refreshControl)
        populateAnnouncements()
        tableView.dataSource = self
        tableView.delegate = self
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(false)
        authenticate()
    }
    
    @objc func handleRefresh() {
        populateAnnouncements()
        refreshControl.endRefreshing()
        
    }
    
    
    
    // REPLACED BY DATABASE CALL, will want to fetch a sorted array by date
    func populateAnnouncements() {
        
    
    
        var ref: DatabaseReference!
        ref = Database.database().reference()
        
        ref.child("announcements").child("slots").observeSingleEvent(of: .value, with: { (snapshot) in
            var announcements : [Announcement] = []
            
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "M/d/yyyy h:mm a"
            
            let value = snapshot.value as? NSDictionary
            for (key, slot) in value! {
                let announcement_data = slot as? NSDictionary
                let database_name = key as! String
                let announcement = Announcement(database_name: database_name, message: announcement_data?["content"] as! String, datetime: dateFormatter.date(from: announcement_data?["timestamp"] as! String)!, author: nil, picture: nil)
                
                announcements.append(announcement)
            }
            
            self.announcements = announcements.sorted(by: { $0.datetime > $1.datetime})
            self.tableView.reloadData()
            
        })
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.announcements.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    
        let cell = tableView.dequeueReusableCell(withIdentifier: "AnnouncementCell", for: indexPath) as! AnnouncementCell
        
        cell.setAnnouncment(announcement: announcements[indexPath.row])
        return cell
    }

    
    func authenticate() {
           
           if Auth.auth().currentUser != nil {
               performSegue(withIdentifier: "userToAdminAnnouncements", sender: nil)
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
