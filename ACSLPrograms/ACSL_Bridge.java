package ACSLPrograms;
import java.util.*;
public class ACSL_Bridge {
	public static int [] runningTotal = {0,0,0,0};
	public static boolean oneHasWon, twoHasWon = false;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		Map<String, Integer> cardVal = new HashMap<String, Integer>();
		cardVal.put("T", 30);
		cardVal.put("H", 30);
		cardVal.put("S", 30);
		cardVal.put("C", 20);
		cardVal.put("D", 20);
		for (int i=0;i<5;i++){
			String input = in.nextLine();
			String inData[] = input.split(",");
			int currTeam = Integer.parseInt(inData[0]);
			int bid = Integer.parseInt(inData[1]);
			int amtWon = Integer.parseInt(inData[2]);
			int val = cardVal.get(inData[3]);
			if (amtWon>=bid+6){
				if(currTeam==1)
					oneHasWon = true;
				else
					twoHasWon = true;
				runningTotal[2*currTeam-2] += bid*val;
				if (inData[3].equals("T")){
					runningTotal[2*currTeam-2]+=10;
				}
				runningTotal[2*currTeam-1] += (amtWon-bid-6)*val;
			}
			else {
				int underticks = Math.abs(amtWon-bid-6);
				if (currTeam == 1){
					if (oneHasWon)
						runningTotal[-1*currTeam + 3] += 100*underticks;
					else
						runningTotal[-1*currTeam + 3] += 50*underticks;
				}
				else{
					if (twoHasWon)
						runningTotal[-1*currTeam + 3] += 100*underticks;
					else
						runningTotal[-1*currTeam + 3] += 50*underticks;
				}
			}
			String output = "";
			for (int x: runningTotal){
				output += x + ",";
			}
			System.out.println(output.substring(0,output.length()-1));
			if(runningTotal[0] >= 100 || runningTotal[2]>= 100){
				runningTotal[0] = 0;
				runningTotal[2] = 0;
				oneHasWon = false;
				twoHasWon = false;
			}
		}
	}
}