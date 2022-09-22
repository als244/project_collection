//
//  QuestionCell.swift
//  YaleSpringFling
//
//  Created by Andrew Sheinberg on 9/22/19.
//

import UIKit

class QuestionCell: UITableViewCell {

    
    var questEdit : (() -> Void)? = nil
    var questDelete : (() -> Void)? = nil
    
    @IBOutlet weak var questionLabel: UILabel!
    
    
    @IBAction func tapEdit(_ sender: Any) {
        if let tapEdit = self.questEdit {
            tapEdit()
        }
    }
    
    @IBAction func tapDelete(_ sender: Any) {
        if let tapDelete = self.questDelete {
            tapDelete()
        }
    
    }
    @IBOutlet weak var carrot: UIImageView!
    func setQuestion(question : String){
        questionLabel.text = question
    }
    
    func setCarrot(carrot_type : String){
        carrot.image = UIImage(named: carrot_type)
    }
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }

}
