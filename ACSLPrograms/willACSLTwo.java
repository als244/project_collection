package ACSLPrograms;

import java.util.*;
public class willACSLTwo {
	public static void main (String args[]){
	for (int ee=0;ee<250;ee++){
		Scanner in = new Scanner(System.in);
		String input = in.nextLine();
		boolean leading=false;
		if (Integer.parseInt(input.substring(0, 1))==0){
			System.out.print("0 ");
			rec(new StringBuilder(input).reverse().toString(),0);
				
			}
		else{rec(input,0);
		}
	
	
	}
	}
	public static void rec(String a, int c){
		int i=1;
		if (a.length() ==0){
			System.out.println();
			return;
		}
		while(Integer.parseInt(a.substring(0,i))<=c){
			i++;
			if (i==a.length()+1){
				System.out.println();
				return;
			}
		}
		
		c =Integer.parseInt(a.substring(0,i));
		System.out.print(c+" ");
		a=a.substring(i);
		
		rec(new StringBuilder(a).reverse().toString(),c);
	}
		
}