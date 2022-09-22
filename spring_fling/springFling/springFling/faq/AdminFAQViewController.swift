//
//  FAQViewController.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 11/6/19.
//

import UIKit
import Firebase

class AdminFAQViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    
    @IBOutlet weak var tableView: UITableView!
    
    
    var q_a_items = [Q_A]()
    var q_a_selected_section : Int? = nil
    let refreshControl = UIRefreshControl()
    
    @IBAction func toAdd(_ sender: Any) {
        performSegue(withIdentifier: "toContent", sender: nil)
    
    }
    override func viewDidLoad() {
        
        super.viewDidLoad()
        authenticate()
        refreshControl.addTarget(self, action: #selector(handleRefresh), for: .valueChanged)
        tableView.addSubview(refreshControl)
        tableView.dataSource = self
        tableView.delegate = self
        populate_q_a_arr()
        
       
        
    }   // Do any additional setup after loading the view.
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(false)
        authenticate()
        populate_q_a_arr()
    }
    
    @objc func handleRefresh() {
        populate_q_a_arr()
        refreshControl.endRefreshing()
        
    }
    
    func populate_q_a_arr() {
        var ref: DatabaseReference!
        ref = Database.database().reference()
        
        ref.child("faq").child("slots").observeSingleEvent(of: .value, with: { (snapshot) in
            // Get all questions and answers
            
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "M/d/yyyy h:mm a"
                                  
            var q_a_items : [Q_A] = []
            let value = snapshot.value as? NSDictionary
            for (key, slot) in value! {
                let q_a_data = slot as? NSDictionary
                let q_a = Q_A(database_name: key as! String, question: q_a_data?["question"] as! String, answer: q_a_data?["answer"] as! String, timestamp: dateFormatter.date(from: q_a_data?["timestamp"] as! String)!)
                q_a_items.append(q_a)
            }
            
            self.q_a_items = q_a_items.sorted(by: { $0.timestamp < $1.timestamp})
            self.tableView.reloadData()

        })
    }
    
    func numberOfSections(in tableView: UITableView) -> Int {
        return q_a_items.count
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        if q_a_items[section].isOpen{
            return 2
        }
        return 1
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        if indexPath.row == 0{
            
            // MIGHT NEED A GUARD HERE
            let cell = tableView.dequeueReusableCell(withIdentifier: "QuestionCell") as! QuestionCell
            cell.setQuestion(question: q_a_items[indexPath.section].question)
            
            
            cell.questEdit = {
            
                self.q_a_selected_section = indexPath.section
                self.performSegue(withIdentifier: "toContent", sender: nil)
                self.q_a_selected_section = nil
                
            }
            
            cell.questDelete = {
                
                self.q_a_selected_section = indexPath.section
                
                let alert = UIAlertController(title: "Delete Announcement", message: "Are you sure you want to permanently delete the announcement?", preferredStyle: .actionSheet)
                
                let DeleteAction = UIAlertAction(title: "Delete", style: .destructive, handler: self.handleDelete)
                let CancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: self.cancelDelete)
                
                alert.addAction(DeleteAction)
                alert.addAction(CancelAction)
                
                self.present(alert, animated: true, completion: nil)
                
                         
            }
                
            return cell
        } else{
            let cell = tableView.dequeueReusableCell(withIdentifier: "AnswerCell") as! AnswerCell
            cell.setAnswer(answer: q_a_items[indexPath.section].answer)
            return cell
        }
    }
    
    func handleDelete(alertAction: UIAlertAction!){
        if let deleteIndex = self.q_a_selected_section{

        tableView.beginUpdates()
                   
        let cell = self.q_a_items[deleteIndex]
                   
                   var ref: DatabaseReference!
                   ref = Database.database().reference()
               ref.child("faq").child("slots").child(cell.database_name).removeValue()
                   
        self.q_a_items.remove(at: deleteIndex)
        tableView.deleteSections([deleteIndex] as IndexSet, with: .left)
                   
        self.q_a_selected_section = nil
                   
        tableView.endUpdates()
        populate_q_a_arr()
    }
}
    
    func cancelDelete(alertAction: UIAlertAction!) {
        self.q_a_selected_section = nil
    }
    
    // the carrot stuff needs to be more stable...?
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        if q_a_items[indexPath.section].isOpen {
            let qIndPath = IndexPath(row: 0, section: indexPath.section)
            let cell = tableView.cellForRow(at: qIndPath) as! QuestionCell
            q_a_items[qIndPath.section].isOpen = false
            //let sections = IndexSet.init(integer: indexPath.section)
            //tableView.reloadS()
            
            UIView.transition(with: tableView, duration: 0.1, options: .transitionCrossDissolve, animations: {self.tableView.reloadData()}, completion: nil)
            cell.setCarrot(carrot_type: "showMoreCarrot")
            
        } else{
            let qIndPath = IndexPath(row: 0, section: indexPath.section)
            let cell = tableView.cellForRow(at: qIndPath) as! QuestionCell
            q_a_items[qIndPath.section].isOpen = true
            //let sections = IndexSet.init(integer: indexPath.section)
            //tableView.reloadData()
            
            UIView.transition(with: tableView, duration: 0.1, options: .transitionCrossDissolve, animations: {self.tableView.reloadData()}, completion: nil)
            
            cell.setCarrot(carrot_type: "revealLessCarrot")
        }
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    guard let contentVC = segue.destination as? ContentFAQViewController else {return}
        
        if self.q_a_selected_section == nil {
                   contentVC.q_a = nil
               } else {
            let q_a = q_a_items[q_a_selected_section!]
               contentVC.q_a = q_a
               }
    }
        
        
        
        
    func authenticate() {
        
        if Auth.auth().currentUser == nil {
            performSegue(withIdentifier: "adminToUserFAQ", sender: nil)
        }
    }
}

