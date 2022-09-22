//
//  LineupViewController.swift
//  Yale Spring Fling
//
//  Created by Alexis Dornan on 10/12/19.
//

import UIKit
import Firebase
import FirebaseStorage

class LineupViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {

    @IBOutlet weak var tableView: UITableView!
    
    private var lineupItems = [LineupSlot]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "Schedule"
    
        populateLineup()
        tableView.dataSource = self
        tableView.delegate = self
        
        let textAttributes = [NSAttributedString.Key.foregroundColor:UIColor.white]
        navigationController?.navigationBar.titleTextAttributes = textAttributes
        
        // Do any additional setup after loading the view.
    }
    
    override func viewWillAppear(_ animated: Bool) {
        
    }
    
    func populateLineup() {
        var ref: DatabaseReference!
        ref = Database.database().reference()
        
        

        ref.child("lineup").child("slots").observeSingleEvent(of: .value, with: { (snapshot) in
            // Get all questions and answers
            var lineupItems : [LineupSlot] = []
            
            
            let value = snapshot.value as? NSDictionary
            for (key, slot) in value! {
                
        
                let lineup_slot = slot as? NSDictionary
                
                let lineupItem = LineupSlot(database_name: key as! String, name: lineup_slot?["name"] as! String, picture: lineup_slot?["picture"] as? String, bio: lineup_slot?["bio"] as? String, spotifyLink: lineup_slot?["spotify_link"] as? String, startTime: lineup_slot?["start_time"] as! String, endTime: lineup_slot?["end_time"] as! String)
              
                lineupItems.append(lineupItem)
               
            }
            
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "hh:mm"
            self.lineupItems = lineupItems.sorted(by: { dateFormatter.date(from: $0.startTime)! < dateFormatter.date(from: $1.startTime)!})
            self.tableView.reloadData()

        })
        
        
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        guard let detailVC = segue.destination as? DetailsViewController else {return}
        let lineupSlot = lineupItems[tableView.indexPathForSelectedRow!.row]
        tableView.deselectRow(at: tableView.indexPathForSelectedRow!, animated: true)
        detailVC.lineupSlot = lineupSlot
    }

    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.lineupItems.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "LineupCell", for: indexPath) as! LineupCell
        
        cell.setLineupSlot(slot : self.lineupItems[indexPath.row])
        
        return cell
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
