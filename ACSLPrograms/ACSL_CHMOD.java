// Barrington High School
// Andrew Sheinberg
// Senior Division 3
package ACSLJuniorYear;
import java.util.*;
public class ACSL_CHMOD {
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		for (int a=0;a<5;a++){
			String input = in.nextLine();
			String [] strIn = input.split(",");
			int [] values = new int[4];
			for (int i=0;i<4;i++){
				values[i]=Integer.parseInt(strIn[i]);
			}
			System.out.println(getBinary(values) + "and " + getLetters(values, getBinary(values)));
		}
	}
	public static String getBinary(int [] x){
		String binary="";
		for (int i=1;i<4;i++){
			if (Integer.toBinaryString(x[i]).length()!=3)
				for (int j=3-Integer.toBinaryString(x[i]).length();j>0;j--){
					binary+="0";
				}
			binary+=Integer.toBinaryString(x[i]) + " ";
		}
		return binary;
	}
	public static String getLetters(int [] x, String s){
		String orig="";
		int modifier = x[0];
		s=s.replaceAll("\\s+","");
		for (int i=0;i<s.length();i++){
			if (i==3 || i==6)
				orig+=" ";
			if (modifier==1 && i==2 && s.charAt(i)=='1' || (modifier==2 && i==5 && s.charAt(i)=='1'))
				orig+='s';
			else if(modifier==4 && i==8 && s.charAt(i)=='1')
				orig+='t';
			else{
				if (s.charAt(i)=='0')
					orig+="-";
				else{
					if(i%3==0)
						orig+="r";
					else if(i%3==1)
						orig+="w";
					else
						orig+="x";
				}
			}
		}
		return orig;
	}
}