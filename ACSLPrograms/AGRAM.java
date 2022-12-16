package ACSLSeniorYear;
import java.util.Scanner;
public class AGRAM {
	public static void main(String[] args) {
		for (int i=0; i<5;i++){
			Scanner in = new Scanner(System.in);
			String input = in.nextLine(), toReturn="";
			String [] arr = input.split(", ");
			int lowestBest = 14, lowSuit = 14, lowVal=14, bestSuit=4;
			boolean best = false, secBest=false;
			String [] ties = {"C", "D", "H", "S"};
			String [] values = {"A", "2", "3", "4", "5", "6", "7", "8", "9" ,"T", "J", "Q", "K"};
			int origVal = java.util.Arrays.asList(values).indexOf(arr[0].substring(0,1)) + 1;
			for (int a = 1; a<6; a++){
				int newVal = java.util.Arrays.asList(values).indexOf(arr[a].substring(0,1)) + 1;
				if (arr[0].substring(1).equals(arr[a].substring(1)) && newVal < lowestBest && newVal > origVal){
					lowestBest = newVal;
					toReturn = arr[a];
					best = true;
				}
				if (!best){ 
					if (arr[0].substring(1).equals(arr[a].substring(1)) && newVal < lowSuit){
						lowSuit = newVal;
						toReturn = arr[a];
						secBest = true;
					}
					if (!secBest){	
						if (newVal < lowVal){
							lowVal = newVal;
							bestSuit = java.util.Arrays.asList(ties).indexOf(arr[a].substring(1));
							toReturn = arr[a];
						}
						else if (newVal==lowVal){
							if (java.util.Arrays.asList(ties).indexOf(arr[a].substring(1)) < bestSuit){
								bestSuit = java.util.Arrays.asList(ties).indexOf(arr[a].substring(1));
								toReturn = arr[a];
							}
						}
					}	
				}	
			}
			System.out.println(toReturn);
		}
	} 
}