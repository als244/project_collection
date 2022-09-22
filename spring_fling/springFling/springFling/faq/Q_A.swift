//
//  Q_A.swift
//  YaleSpringFling
//
//  Created by Andrew Sheinberg on 9/22/19.


import UIKit
class Q_A {
    
    var database_name: String
    var question: String
    var answer : String
    var isOpen : Bool
    var timestamp: Date
    
    init(database_name: String, question: String, answer: String, timestamp: Date){
        self.database_name = database_name
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.isOpen = false
    }
}


