//
//  FAQViewController.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 10/13/19.
//

import UIKit
import Firebase

class FAQViewController: UIViewController, UITableViewDataSource, UITableViewDelegate {
    
    @IBOutlet weak var tableView: UITableView!
    
    
    var q_a_items = [Q_A]()
    let refreshControl = UIRefreshControl()
    
    
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
                
            return cell
        } else{
            let cell = tableView.dequeueReusableCell(withIdentifier: "AnswerCell") as! AnswerCell
            cell.setAnswer(answer: q_a_items[indexPath.section].answer)
            return cell
        }
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
    
    func authenticate() {
        
        if Auth.auth().currentUser != nil {
            performSegue(withIdentifier: "userToAdminFAQ", sender: nil)
        }
    }
}
