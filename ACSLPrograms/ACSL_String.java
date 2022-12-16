// Andrew Sheinberg
// Barrington High School RI
// Senior Division 3
package ACSLJuniorYear;
import java.util.*;
import java.math.RoundingMode;
import java.text.*;
public class ACSL_String {
	private static String sign;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		for (int a=0;a<5;a++){
			String input = in.nextLine();
			String [] strIn = input.split(",");
			double [] values = new double[3];
			if (strIn[0].substring(0,1).equals("+") || strIn[0].substring(0,1).equals("-")){
				sign=strIn[0].substring(0,1);
				strIn[0]=strIn[0].substring(1);
			}
			else
				sign="";	
			for (int i=0;i<3;i++){
				values[i]=Double.parseDouble(strIn[i]);
			}
			System.out.println(sign + format(values));
		}
	}
	public static String format(double [] x){
		String baseStr="";
		double decimalPlaces= x[2];
		double hashBeforeDecimal; 
		if (! sign.equals(""))
			hashBeforeDecimal = x[1]-x[2] - 2;
		else
			hashBeforeDecimal = x[1]-x[2] - 1;
		if (x[2]==0)
			hashBeforeDecimal++;
		int lengthBeforeDecimal = (x[0]+"").indexOf(".");
		if (hashBeforeDecimal<lengthBeforeDecimal){
			for (double i=0;i<hashBeforeDecimal;i++){
				baseStr+="#";
			}
			baseStr+=".";
			for (double i=0;i<decimalPlaces;i++){
				baseStr+="#";
			}
			return baseStr;
		}
		String formatterStr = "";
		for (double i=0;i<hashBeforeDecimal;i++){
			formatterStr+="0";
		}
		if (x[2]!=0)
			formatterStr+=".";
		for (double i=0;i<decimalPlaces;i++){		
			formatterStr+="0";
		}
		DecimalFormat formatter = new DecimalFormat(formatterStr);
		formatter.setRoundingMode(RoundingMode.HALF_UP);
		String ans = formatter.format(x[0]);
		String addInFront="";
		if (hashBeforeDecimal>lengthBeforeDecimal){
			for (int i=0;i<hashBeforeDecimal-lengthBeforeDecimal;i++){
				addInFront+="#";
			}
			ans=ans.substring((int)(hashBeforeDecimal-lengthBeforeDecimal));
		}
		return addInFront + ans;
	}
}