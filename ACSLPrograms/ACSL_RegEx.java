// Andrew Sheinberg
// Barrington High School
// Senior Division 3
package ACSLJuniorYear;
import java.util.*;
public class ACSL_RegEx {
	private static String[] strings= new String[10];
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		strings=(in.nextLine()).split(",");
		for (int i=0;i<10;i++){
			if (strings[i].equals("#"))
				strings[i]="";
		}
		for (int a=0;a<5;a++){
			System.out.println(match(in.nextLine()));
		}
	}
	public static String match(String s){
		String ans="";
		boolean isMatch=false;
		for (String test: strings){
			if (test.matches(s)){
				if (test.equals(""))
					ans+="#"+",";
				else
					ans+=test+",";
				isMatch=true;
			}
		}
		if (!isMatch)
			return "NONE";
		return ans.substring(0,ans.length()-1);
	}
}