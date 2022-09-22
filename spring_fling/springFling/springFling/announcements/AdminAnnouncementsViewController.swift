//
//  AdminAnnouncementsViewController.swift
//  springFling
//
//  Created by Andrew Sheinberg on 11/4/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit
import Foundation
import Firebase

class AdminAnnouncementsViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {

    @IBOutlet weak var tableView: UITableView!

    
    @IBAction func editTap(_ sender: Any) {
    }
    
    
    @IBAction func tapAdd(_ sender: Any) {
        performSegue(withIdentifier: "toContent", sender: nil)
    
    }

    
    var announcements  = [Announcement]()
    var admin : Bool?
    var deleteAnnouncementPath : IndexPath? = nil
    let refreshControl = UIRefreshControl()
    
    override func viewDidLoad() {
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
        populateAnnouncements()
        tableView.reloadData()
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
                print(database_name)
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

    func tableView(_ tableView: UITableView, editingStyleForRowAt indexPath: IndexPath) -> UITableViewCell.EditingStyle {
        return .delete
    }
    
    func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        if editingStyle == .delete {
           
            deleteAnnouncementPath = indexPath
            
            let alert = UIAlertController(title: "Delete Announcement", message: "Are you sure you want to permanently delete the announcement?", preferredStyle: .actionSheet)
            
             let DeleteAction = UIAlertAction(title: "Delete", style: .destructive, handler: handleDelete)
             let CancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: cancelDelete)
            
            alert.addAction(DeleteAction)
            alert.addAction(CancelAction)
            
            self.present(alert, animated: true, completion: nil)
            
            
            
        }
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        guard let contentVC = segue.destination as? ContentAnnouncementViewController else {return}
        
        if tableView.indexPathForSelectedRow == nil {
            contentVC.announcement = nil
        } else {
        let announcement = announcements[tableView.indexPathForSelectedRow!.row]
        tableView.deselectRow(at: tableView.indexPathForSelectedRow!, animated: true)
        contentVC.announcement = announcement
        }
    }
    
    
    func handleDelete(alertAction: UIAlertAction!) {
        if let indexPath = deleteAnnouncementPath {
            tableView.beginUpdates()
            
            let cell = announcements[indexPath.row]
            
            var ref: DatabaseReference!
            ref = Database.database().reference()
        ref.child("announcements").child("slots").child(cell.database_name).removeValue()
            
            announcements.remove(at: indexPath.row)
            tableView.deleteRows(at: [indexPath], with: .fade)
            
            deleteAnnouncementPath = nil
            
            tableView.endUpdates()
          
        }
        
    }
    
    func cancelDelete(alertAction: UIAlertAction!) {
        deleteAnnouncementPath = nil
    }
    
    func authenticate() {
        
        if Auth.auth().currentUser == nil {
            performSegue(withIdentifier: "adminToUserAnnouncements", sender: nil)
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
