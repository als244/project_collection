package ACSLJuniorYear;
import java.util.*;
public class ACSL_ABC {
	private static String[][] board;
	private static String[][] simpBoard;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		for (int i = 0; i < 5; i++) {
			String input = in.nextLine();
			String[] inputArr = input.split(",");
			initialize(inputArr);
			fillObvious();
			deduce();
			System.out.println(printAns());
		}
	}
	public static void initialize(String[] x) {
		int[] y = new int[4];
		board = new String[6][6];
		for (int i = 0; i < 4; i++) {
			y[i] = Integer.parseInt(x[i]);
			board[(y[i] - 1) / 6][y[i] - 1 - (((y[i] - 1) / 6) * 6)] = "*";
		}
		int[] indicaters = new int[Integer.parseInt(x[4])];
		int cnt = 0, row = 0, col = 0;
		for (int i = 6; i < 7+(2*(Integer.parseInt(x[4])-1)); i += 2) {
			indicaters[cnt] = Integer.parseInt(x[i]);
			row = ((indicaters[cnt] - 1) / 6);
			col = indicaters[cnt] - 1 - (row * 6);
			if (row == 0) {
				if (board[row + 1][col] == null)
					board[row + 1][col] = x[i - 1];
				else
					board[row + 2][col] = x[i - 1];
			}
			if (row == 5) {
				if (board[row - 1][col] == null)
					board[row - 1][col] = x[i - 1];
				else
					board[row - 2][col] = x[i - 1];
			}
			if (col == 0) {
				if (board[row][col + 1] == null)
					board[row][col + 1] = x[i - 1];
				else
					board[row][col + 2] = x[i - 1];
			}
			if (col == 5) {
				if (board[row][col - 1] == null)
					board[row][col - 1] = x[i - 1];
				else
					board[row][col - 2] = x[i - 1];
			}
			cnt++;
		}
		int rcnt = 0, ccnt = 0;
		simpBoard = new String[4][4];
		for (int i = 1; i < 5; i++) {
			for (int j = 1; j < 5; j++) {
				if (board[i][j] != null)
					simpBoard[rcnt][ccnt] = board[i][j];
				ccnt++;
			}
			rcnt++;
			ccnt = 0;
		}
	}
	public static void fillObvious() {
		int cnt = 0;
		boolean hasA = false, hasB = false, hasC = false;
		for (int z = 0; z < 10; z++) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					if (simpBoard[i][j] == null)
						cnt++;
					else {
						if (simpBoard[i][j].equals("A"))
							hasA = true;
						if (simpBoard[i][j].equals("B"))
							hasB = true;
						if (simpBoard[i][j].equals("C"))
							hasC = true;
					}
				}
				if (cnt <= 1) {
					for (int a = 0; a < 4; a++) {
						if (simpBoard[i][a] == null) {
							if (hasA && hasB) {
								simpBoard[i][a] = "C";
								break;
							}
							if (hasA && hasC) {
								simpBoard[i][a] = "B";
								break;
							}
							if (hasB && hasC) {
								simpBoard[i][a] = "A";
								break;
							}
						}
					}
				}
				cnt = 0;
				hasA = false;
				hasB = false;
				hasC = false;
			}
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					if (simpBoard[j][i] == null)
						cnt++;
					else {
						if (simpBoard[j][i].equals("A"))
							hasA = true;
						if (simpBoard[j][i].equals("B"))
							hasB = true;
						if (simpBoard[j][i].equals("C"))
							hasC = true;
					}
				}
				if (cnt <= 1) {
					for (int a = 0; a < 4; a++) {
						if (simpBoard[a][i] == null) {
							if (hasA && hasB) {
								simpBoard[a][i] = "C";
								break;
							}
							if (hasA && hasC) {
								simpBoard[a][i] = "B";
								break;
							}
							if (hasB && hasC) {
								simpBoard[a][i] = "A";
								break;
							}
						}
					}
				}
				cnt = 0;
				hasA = false;
				hasB = false;
				hasC = false;
			}
		}
	}
	public static void deduce() {
		String known1 = "", known2 = "";
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (simpBoard[i][j] == null) {
					for (int a = 0; a < 4; a++) {
						if (simpBoard[i][a] != null) {
							if (!simpBoard[i][a].equals("*"))
								known1 = simpBoard[i][a];
						}
						if (simpBoard[a][j] != null) {
							if (!simpBoard[a][j].equals("*"))
								known2 = simpBoard[a][j];
						}
					}
					if (!known1.equals("") && !known2.equals("")) {
						if (!known1.equals(known2)) {
							if (known1.equals("A") && known2.equals("B")) {
								simpBoard[i][j] = "C";
								break;
							}
							if (known1.equals("B") && known2.equals("A")) {
								simpBoard[i][j] = "C";
								break;
							}
							if (known1.equals("A") && known2.equals("C")) {
								simpBoard[i][j] = "B";
								break;
							}
							if (known1.equals("C") && known2.equals("A")) {
								simpBoard[i][j] = "B";
								break;
							}
							if (known1.equals("B") && known2.equals("C")) {
								simpBoard[i][j] = "A";
								break;
							}
							if (known1.equals("C") && known2.equals("B")) {
								simpBoard[i][j] = "A";
								break;
							}
						}
					}
					known1 = "";
					known2 = "";
				}
			}
			fillObvious();
		}
	}
	public static String printAns(){
		String ans="";
		for (int i=0;i<4;i++){
			for (int j=0;j<4;j++){
				if (simpBoard[i][j].equals("A") || simpBoard[i][j].equals("B") || simpBoard[i][j].equals("C"))
					ans+=simpBoard[i][j];
			}
		}
		return ans;
	}
}