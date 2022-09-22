//
//  lineupItem.swift
//  Yale Spring Fling
//
//  Created by Alexis Dornan on 10/10/19.
//


class LineupSlot{
    
    // maybe make time a date..?
    var startTime: String
    var endTime: String
    
    var name: String
    var picture : String?
       
    var bio : String?
       
    var spotifyLink : String?
    var database_name: String

    init(database_name: String, name: String, picture: String?, bio : String?, spotifyLink: String?, startTime: String, endTime:String){
        
        self.database_name = database_name
        self.name = name
        self.picture = picture
        self.bio = bio
        self.spotifyLink = spotifyLink
        self.startTime = startTime
        self.endTime = endTime
    }
    

}
