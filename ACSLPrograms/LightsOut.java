// Andrew Sheinberg
// Barrington High School
// Senior Division 5
package ACSLSeniorYear;
import java.util.*;
public class LightsOut {
	private static boolean[][] origBoard = new boolean[8][8];
	private static boolean[][] newBoard = new boolean[8][8];
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		for (int z = 0; z<6; z++){
			String line = in.nextLine();
			line = line.replace(" ", "");
			String strToBoard = "";
			for (int x = 0; x< line.length(); x++){
				strToBoard += hexToBinary(line.substring(x, x+1));
			}
			if (z==0)
				strToOrigBoard(strToBoard);
			else{
				strToNewBoard(strToBoard);
				System.out.println(testPoints());
				origBoard = newBoard;
				newBoard = new boolean[8][8];
			}
		}		
	}
	public static String hexToBinary(String hex) {
	    int i = Integer.parseInt(hex, 16);
	    String bin = Integer.toBinaryString(i);
	    while (bin.length() < 4){
	    	bin = "0" + bin;
	    }
	    return bin;
	} 
	public static void strToOrigBoard(String str){
		int cnt = 0;
		for (int row = 7; row >=0 ; row--){
			for (int col = 0; col <8; col++){
				if (str.substring(cnt, cnt+1).equals("1"))
					origBoard[row][col] = true;
				else
					origBoard[row][col] = false;
				cnt++;
			}
		}
	}
	public static void strToNewBoard(String str){
		int cnt = 0;
		for (int row = 7; row >=0 ; row--){
			for (int col = 0; col<8; col++){
				if (str.substring(cnt, cnt+1).equals("1"))
					newBoard[row][col] = true;
				else
					newBoard[row][col] = false;
				cnt++;
			}
		}
	}
	public static void flipTiles(int row, int col){
		for (int v = -1; v<2; v++){
			for (int h = -1; h<2; h++){
				try{
					origBoard[row + h][col + v] = !origBoard[row + h][col + v];
				}
				catch(ArrayIndexOutOfBoundsException exception) {
				}
			}
		}
		for (int r = -2; r<3; r+=4){
			try{
				origBoard[row + r][col] = !origBoard[row + r][col];
			}
			catch(ArrayIndexOutOfBoundsException exception) {
			}
		}
		for (int  c= -2; c<3; c+=4){
			try{
				origBoard[row][col + c] = !origBoard[row][col + c];
			}
			catch(ArrayIndexOutOfBoundsException exception) {
			}
		}
	}
	public static boolean doesMatch(){
		for (int i = 0; i<8; i++){
			for (int j = 0; j<8; j++){
				if (origBoard[i][j] != newBoard[i][j])
					return false;
			}
		}
		return true;
	}
	public static String testPoints(){
		for (int i = 0; i<8; i++){
			for (int j = 0; j<8; j++){
				flipTiles(i, j);
				if (doesMatch())
					return "" + (8-i) + (j+1);	
				else
					flipTiles(i, j);
			}
		}
		return "";
	}
}