//
//  ContentFAQViewController.swift
//  springFling
//
//  Created by Andrew Sheinberg on 11/6/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit
import Firebase
import Foundation

class ContentFAQViewController: UIViewController, UIScrollViewDelegate {
    
    
    @IBOutlet weak var topLabel: UILabel!
    
    @IBOutlet weak var questionText: UITextView!
    
    @IBOutlet weak var answerText: UITextView!
    
    var q_a: Q_A!
    
    @IBAction func submit(_ sender: Any) {
        
        if (questionText.text == "" && answerText.text == "") {
            self.dismiss(animated: true, completion: nil)
            return
        }
        
        var ref: DatabaseReference!
        ref = Database.database().reference()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "M/d/yyyy h:mm a"
               
        
        if q_a != nil {
            ref.child("faq").child("slots").child(q_a.database_name).setValue(["question":questionText.text!, "answer":answerText.text!, "timestamp":dateFormatter.string(from: q_a.timestamp)])
            
        } else{
            guard let key = ref.child("faq").child("slots").childByAutoId().key else { return }
            
            let time = dateFormatter.string(from: Date())
            ref.child("faq").child("slots").child(key).setValue(["question":questionText.text!, "answer": answerText.text!, "timestamp":time])
            
        }
        
          self.dismiss(animated: true, completion: nil)
    
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.hideKeyboardWhenTappedAround()
        questionText.textContainerInset = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        answerText.textContainerInset = UIEdgeInsets(top: 10, left: 10, bottom: 10, right: 10)
        
        if q_a != nil {
            topLabel.text = "Edit"
            questionText.text = q_a.question
            answerText.text = q_a.answer
        } else{
            topLabel.text = "Add"
            
        }
        
        
        
    }
    
    func scrollViewWillBeginDragging(_ scrollView: UIScrollView) {
        self.questionText.resignFirstResponder()
        self.answerText.resignFirstResponder()
    }
}
