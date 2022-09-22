//
//  ResourcesViewController.swift
//  Yale Spring Fling
//
//  Created by Andrew Sheinberg on 9/29/19.
//

import UIKit
import Firebase

class ResourcesViewController: UIViewController, UITableViewDataSource, UITableViewDelegate  {
    
    @IBOutlet weak var tableView: UITableView!
  
    private var resources: [Resource] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        
        navigationController?.navigationBar.prefersLargeTitles = true
        self.title = "Resources"
        resources = createResources()
        
        if Auth.auth().currentUser == nil {
        resources.append(Resource(name: "Admin Login", phoneNumber: nil, link: nil, picture: nil))
        } else{
            resources.append(Resource(name: "Sign Out", phoneNumber: nil, link: nil, picture: nil))
        }
        
        
        // Do any additional setup after loading the view.
        tableView.dataSource = self
        tableView.delegate = self
    }
    

    override func viewWillAppear(_ animated: Bool) {
        authenticate()
        tableView.reloadData()
    }
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
    
        // deselect the selected row if any
        let selectedRow: IndexPath? = tableView.indexPathForSelectedRow
        if let selectedRowNotNill = selectedRow {
            tableView.deselectRow(at: selectedRowNotNill, animated: true)
        }
    }
    
    func authenticate() {
        if Auth.auth().currentUser == nil {
            self.resources[resources.count - 1].name = "Admin Login"
               } else {
            self.resources[resources.count - 1].name = "Sign Out"
        }
    }
    // WILl BE REPLACED BY DATABASE CALL
    func createResources() -> [Resource] {
        return [Resource(name: "Call Yale Police", phoneNumber: "tel:2034324400", link: nil, picture:nil),
                Resource(name: "Call 911", phoneNumber: "tel:911", link: nil, picture:nil), Resource(name: "Spring Fling Website", phoneNumber: nil, link: "https://springfling.yale.edu/", picture: nil) , Resource(name: "Spring Fling Facebook", phoneNumber: nil, link: "https://www.facebook.com/yalespringfling/", picture:nil), Resource(name: "Map", phoneNumber: nil, link: nil, picture: UIImage(named: "oldCampusMap"))]
    }
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return self.resources.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath) as! ResourceCell
        cell.setResource(resource: resources[indexPath.row])
        return cell
    }
    
    // THIS ISN"T GOOD CODE TO PREFORM ACTION AFTER CLICK
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let index = indexPath.row
        if resources[index].picture != nil {
            let imageVC = self.storyboard?.instantiateViewController(withIdentifier: "ResourceImageViewController") as! ResourceImageViewController
            imageVC.resource = resources[index]
            self.navigationController?.pushViewController(imageVC, animated: true)
        }
        else if let link = resources[index].link{
            print(link)
            if let url = URL(string:link) {
                UIApplication.shared.open(url)
            }
        }
        else if let phoneNumber = resources[index].phoneNumber{
            if let numberUrl = URL(string: phoneNumber){
                print(numberUrl)
                UIApplication.shared.open(numberUrl)
            }
        }
        else{
            
            if resources[index].name == "Sign Out" {
                
                print("SIGNING OUT")
                let alertController = UIAlertController(title: nil, message: "Are you sure you want to sign out?", preferredStyle: .actionSheet)
                alertController.addAction(UIAlertAction(title: "Sign Out", style: .destructive, handler: { (_) in
                    self.doSignOut()
                }))
                alertController.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
                present(alertController, animated: true, completion: nil)
                
            } else {
                let storyboard = UIStoryboard(name: "login", bundle: nil)
                    let vc = storyboard.instantiateInitialViewController()!
                
                    self.navigationController?.pushViewController(vc, animated: true)
            }
            
        }
        tableView.deselectRow(at: indexPath, animated: true)
    }
       
    func doSignOut() {
        do {
            try Auth.auth().signOut()
            authenticate()
            tableView.reloadData()
        } catch let error {
            print("Failed to sign out with error..", error)
        }
    }
}
