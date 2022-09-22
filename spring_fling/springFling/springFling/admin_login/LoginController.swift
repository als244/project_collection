//
//  LoginController.swift
//  springFling
//
//  Created by Andrew Sheinberg on 11/4/19.
//  Copyright © 2019 Andrew Sheinberg. All rights reserved.
//

import UIKit
import Firebase

class LoginController: UIViewController {
    
    // MARK: - Properties

    @IBOutlet weak var email: UITextField!
    
    @IBOutlet weak var password: UITextField!
    
    @IBOutlet weak var error_field: UILabel!
    
    // MARK: - Init
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.hideKeyboardWhenTappedAround() 
    }
    
    
    func logInError(error: Error){
        error_field?.text = error.localizedDescription
        email.text = ""
        password.text = ""
    }

    func logUserIn(withEmail email: String, password: String)  {
        Auth.auth().signIn(withEmail: email, password: password) { [weak self] user, error in
            
            guard let strongSelf = self else { return }
            // [START_EXCLUDE]

            if let error = error {
                print(error.localizedDescription)
                strongSelf.logInError(error: error)
            }
            else{
                strongSelf.navigationController?.popViewController(animated: true)
            }
        }
    }
    
    @IBAction func tapLogin(_ sender: Any) {
        
        if let user_email = email.text , let user_password = password.text {
            logUserIn(withEmail: user_email, password: user_password)
        }
        else{
            print("need email and password")
        }
    }
}

extension UIViewController {
    func hideKeyboardWhenTappedAround() {
        let tap: UITapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(UIViewController.dismissKeyboard))
        tap.cancelsTouchesInView = false
        view.addGestureRecognizer(tap)
    }

    @objc func dismissKeyboard() {
        view.endEditing(true)
    }
}
