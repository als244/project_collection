//
//  AnswerCell.swift
//  YaleSpringFling
//
//  Created by Andrew Sheinberg on 9/22/19.
//

import UIKit

class AnswerCell: UITableViewCell {

    @IBOutlet weak var answerLabel: UILabel!
    
    func setAnswer(answer : String){
        answerLabel.text = answer
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
