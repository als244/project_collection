// Andrew Sheinberg
import java.util.*;
public class Quine {
	private static String input;
	private static boolean[][] threeVarBoard;
	private static boolean[][] fourVarBoard;
	private static ArrayList<Integer> threeVarVertCol;
	private static ArrayList<Integer> vertCol;
	private static ArrayList<Integer> threeVarHorizRow;
	private static ArrayList<Integer> horizRow;
	private static ArrayList<Integer> threeVarSquareStart;
	private static ArrayList<Integer> squareStart;
	private static ArrayList<Integer> fourVarBottomBoarder;
	private static ArrayList<Integer> fourVarSideBoarder;
	private static ArrayList<Integer> fourVarTwoDown;
	private static ArrayList<Integer> fourVarTwoAcross;
	private static ArrayList<Integer> vertTwoDown;
	private static ArrayList<String> threeVarAnswers;
	private static ArrayList<String> fourVarAnswers;

	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		for (int i = 0; i < 3; i++) {
			input = in.nextLine();
			initializeThreeVarBoard();
			checkCombosThreeVar();
			threeVarAnswers();
			threeVarRemoval();
			printThreeVarAnswer();
			System.out.println();
		}
		for (int i = 0; i < 2; i++) {
			input = in.nextLine();
			initializeFourVarBoard();
			checkCombosFourVar();
			fourVarAnswers();
			printFourVarAnswer();
			System.out.println();
		}
	}
	public static void initializeThreeVarBoard() {
		threeVarBoard = new boolean[4][2];
		String[] seperatedInput = input.split(",");
		int[] inputAsInt = new int[seperatedInput.length - 1];
		for (int i = 0; i < seperatedInput.length - 1; i++) {
			inputAsInt[i] = Integer.parseInt(seperatedInput[i]);
		}
		for (int i = 0; i < inputAsInt.length; i++) {
			if (inputAsInt[i] == 0)
				threeVarBoard[0][0] = true;
			if (inputAsInt[i] == 1)
				threeVarBoard[1][0] = true;
			if (inputAsInt[i] == 2)
				threeVarBoard[3][0] = true;
			if (inputAsInt[i] == 3)
				threeVarBoard[2][0] = true;
			if (inputAsInt[i] == 4)
				threeVarBoard[0][1] = true;
			if (inputAsInt[i] == 5)
				threeVarBoard[1][1] = true;
			if (inputAsInt[i] == 6)
				threeVarBoard[3][1] = true;
			if (inputAsInt[i] == 7)
				threeVarBoard[2][1] = true;
		}
	}
	public static void initializeFourVarBoard() {
		fourVarBoard = new boolean[4][4];
		String[] seperatedInput = input.split(",");
		int[] inputAsInt = new int[seperatedInput.length - 1];
		for (int i = 0; i < seperatedInput.length - 1; i++) {
			inputAsInt[i] = Integer.parseInt(seperatedInput[i]);
		}
		for (int i = 0; i < inputAsInt.length; i++) {
			if (inputAsInt[i] == 0)
				fourVarBoard[0][0] = true;
			if (inputAsInt[i] == 1)
				fourVarBoard[1][0] = true;
			if (inputAsInt[i] == 2)
				fourVarBoard[3][0] = true;
			if (inputAsInt[i] == 3)
				fourVarBoard[2][0] = true;
			if (inputAsInt[i] == 4)
				fourVarBoard[0][1] = true;
			if (inputAsInt[i] == 5)
				fourVarBoard[1][1] = true;
			if (inputAsInt[i] == 6)
				fourVarBoard[3][1] = true;
			if (inputAsInt[i] == 7)
				fourVarBoard[2][1] = true;
			if (inputAsInt[i] == 8)
				fourVarBoard[0][3] = true;
			if (inputAsInt[i] == 9)
				fourVarBoard[1][3] = true;
			if (inputAsInt[i] == 10)
				fourVarBoard[3][3] = true;
			if (inputAsInt[i] == 11)
				fourVarBoard[2][3] = true;
			if (inputAsInt[i] == 12)
				fourVarBoard[0][2] = true;
			if (inputAsInt[i] == 13)
				fourVarBoard[1][2] = true;
			if (inputAsInt[i] == 14)
				fourVarBoard[3][2] = true;
			if (inputAsInt[i] == 15)
				fourVarBoard[2][2] = true;
		}
	}
	public static void checkCombosThreeVar() {
		threeVarVertCol = new ArrayList<Integer>();
		threeVarHorizRow = new ArrayList<Integer>();
		threeVarSquareStart = new ArrayList<Integer>();
		vertTwoDown = new ArrayList<Integer>();
		for (int i = 0; i < 4; i++) {
			if (threeVarBoard[i][0] && threeVarBoard[i][1])
				threeVarHorizRow.add(i);
		}
		for (int i = 0; i < 2; i++) {
			if (threeVarBoard[0][i] && threeVarBoard[1][i]
					&& threeVarBoard[2][i] && threeVarBoard[3][i])
				threeVarVertCol.add(i);
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 1; j++) {
				if (threeVarBoard[i][j] && threeVarBoard[i][j + 1]
						&& threeVarBoard[i + 1][j]
						&& threeVarBoard[i + 1][j + 1])
					threeVarSquareStart.add(i);
			}
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				if (threeVarBoard[i][j] && threeVarBoard[i + 1][j]) {
					vertTwoDown.add(i);
					vertTwoDown.add(j);
				}
			}
		}
		if (threeVarBoard[3][0] && threeVarBoard[0][0]) {
			vertTwoDown.add(3);
			vertTwoDown.add(0);
		}
		if (threeVarBoard[3][1] && threeVarBoard[0][1]) {
			vertTwoDown.add(3);
			vertTwoDown.add(1);
		}
	}
	public static void checkCombosFourVar() {
		vertCol = new ArrayList<Integer>();
		horizRow = new ArrayList<Integer>();
		squareStart = new ArrayList<Integer>();
		fourVarBottomBoarder = new ArrayList<Integer>();
		fourVarSideBoarder = new ArrayList<Integer>();
		fourVarTwoDown = new ArrayList<Integer>();
		fourVarTwoAcross = new ArrayList<Integer>();
		for (int i = 0; i < 4; i++) {
			if (fourVarBoard[i][0] && fourVarBoard[i][1] && fourVarBoard[i][2]
					&& fourVarBoard[i][3])
				horizRow.add(i);
		}
		for (int i = 0; i < 4; i++) {
			if (fourVarBoard[0][i] && fourVarBoard[1][i] && fourVarBoard[2][i]
					&& fourVarBoard[3][i])
				vertCol.add(i);
		}
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (fourVarBoard[i][j] && fourVarBoard[i][j + 1]
						&& fourVarBoard[i + 1][j] && fourVarBoard[i + 1][j + 1]) {
					squareStart.add(i);
					squareStart.add(j);
				}
			}
		}
		for (int i = 0; i < 3; i++) {
			if (fourVarBoard[3][i] && fourVarBoard[0][i]
					&& fourVarBoard[3][i + 1] && fourVarBoard[0][i + 1])
				fourVarBottomBoarder.add(i);
		}
		for (int i = 0; i < 3; i++) {
			if (fourVarBoard[i][0] && fourVarBoard[i + 1][0]
					&& fourVarBoard[i][3] && fourVarBoard[i + 1][3])
				fourVarSideBoarder.add(i);
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (i == 3) {
					if (fourVarBoard[3][j] && fourVarBoard[0][j]) {
						fourVarTwoDown.add(i);
						fourVarTwoDown.add(j);
					}
				} else if (fourVarBoard[i][j] && fourVarBoard[i + 1][j]) {
					fourVarTwoDown.add(i);
					fourVarTwoDown.add(j);
				}
			}
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (j == 3) {
					if (fourVarBoard[i][3] && fourVarBoard[i][0]) {
						fourVarTwoAcross.add(i);
						fourVarTwoAcross.add(j);
					}
				} else if (fourVarBoard[i][j] && fourVarBoard[i][j + 1]) {
					fourVarTwoAcross.add(i);
					fourVarTwoAcross.add(j);
				}
			}
		}
	}
	public static void threeVarAnswers() {
		threeVarAnswers = new ArrayList<String>();
		for (int i = 0; i < threeVarVertCol.size(); i++) {
			if (threeVarVertCol.get(i) == 0)
				threeVarAnswers.add("a");
			if (threeVarVertCol.get(i) == 1)
				threeVarAnswers.add("A");
		}
		for (int i = 0; i < threeVarSquareStart.size(); i++) {
			if (threeVarSquareStart.get(i) == 0)
				threeVarAnswers.add("b");
			if (threeVarSquareStart.get(i) == 1)
				threeVarAnswers.add("C");
			if (threeVarSquareStart.get(i) == 2)
				threeVarAnswers.add("B");
		}
		if (threeVarBoard[0][0] && threeVarBoard[0][1] && threeVarBoard[3][0]
				&& threeVarBoard[3][1])
			threeVarAnswers.add("c");
		int row;
		int col;
		for (int i = 0; i < vertTwoDown.size(); i += 2) {
			row = vertTwoDown.get(i);
			col = vertTwoDown.get(i + 1);
			if (row == 0 && col == 0)
				threeVarAnswers.add("ab");
			if (row == 1 && col == 0)
				threeVarAnswers.add("aC");
			if (row == 2 && col == 0)
				threeVarAnswers.add("aB");
			if (row == 0 && col == 1)
				threeVarAnswers.add("Ab");
			if (row == 1 && col == 1)
				threeVarAnswers.add("AC");
			if (row == 2 && col == 1)
				threeVarAnswers.add("AB");
			if (row == 3 && col == 0)
				threeVarAnswers.add("ac");
			if (row == 3 && col == 1)
				threeVarAnswers.add("Ac");
		}
		for (int i = 0; i < threeVarHorizRow.size(); i++) {
			if (threeVarHorizRow.get(i) == 0)
				threeVarAnswers.add("bc");
			if (threeVarHorizRow.get(i) == 1)
				threeVarAnswers.add("bC");
			if (threeVarHorizRow.get(i) == 2)
				threeVarAnswers.add("BC");
			if (threeVarHorizRow.get(i) == 3)
				threeVarAnswers.add("Bc");
		}
	}
	public static void threeVarRemoval() {
		if (threeVarAnswers.contains("a")) {
			threeVarAnswers.remove("ab");
			threeVarAnswers.remove("aB");
			threeVarAnswers.remove("aC");
			threeVarAnswers.remove("ac");
		}
		if (threeVarAnswers.contains("A")) {
			threeVarAnswers.remove("Ab");
			threeVarAnswers.remove("AB");
			threeVarAnswers.remove("AC");
			threeVarAnswers.remove("Ac");
		}
		if (threeVarAnswers.contains("b")) {
			threeVarAnswers.remove("ab");
			threeVarAnswers.remove("Ab");
			threeVarAnswers.remove("bC");
			threeVarAnswers.remove("bc");
		}
		if (threeVarAnswers.contains("B")) {
			threeVarAnswers.remove("aB");
			threeVarAnswers.remove("AB");
			threeVarAnswers.remove("BC");
			threeVarAnswers.remove("Bc");
		}
		if (threeVarAnswers.contains("c")) {
			threeVarAnswers.remove("ac");
			threeVarAnswers.remove("Ac");
			threeVarAnswers.remove("bc");
			threeVarAnswers.remove("Bc");
		}
		if (threeVarAnswers.contains("C")) {
			threeVarAnswers.remove("aC");
			threeVarAnswers.remove("AC");
			threeVarAnswers.remove("BC");
			threeVarAnswers.remove("bC");
		}
	}
	public static void fourVarAnswers() {
		fourVarAnswers = new ArrayList<String>();
		for (int i = 0; i < vertCol.size(); i++) {
			if (vertCol.get(i) == 0)
				fourVarAnswers.add("ab");
			if (vertCol.get(i) == 1)
				fourVarAnswers.add("aB");
			if (vertCol.get(i) == 2)
				fourVarAnswers.add("AB");
			if (vertCol.get(i) == 3)
				fourVarAnswers.add("Ab");
		}
		for (int i = 0; i < horizRow.size(); i++) {
			if (horizRow.get(i) == 0)
				fourVarAnswers.add("cd");
			if (horizRow.get(i) == 1)
				fourVarAnswers.add("cD");
			if (horizRow.get(i) == 2)
				fourVarAnswers.add("CD");
			if (horizRow.get(i) == 3)
				fourVarAnswers.add("Cd");
		}
		int row;
		int col;
		for (int i = 0; i < squareStart.size(); i += 2) {
			row = squareStart.get(i);
			col = squareStart.get(i + 1);
			if (row == 0 && col == 0)
				fourVarAnswers.add("ac");
			if (row == 1 && col == 0)
				fourVarAnswers.add("aD");
			if (row == 2 && col == 0)
				fourVarAnswers.add("aC");
			if (row == 0 && col == 1)
				fourVarAnswers.add("Bc");
			if (row == 1 && col == 1)
				fourVarAnswers.add("BD");
			if (row == 2 && col == 1)
				fourVarAnswers.add("BC");
			if (row == 0 && col == 2)
				fourVarAnswers.add("Ac");
			if (row == 1 && col == 2)
				fourVarAnswers.add("AD");
			if (row == 2 && col == 2)
				fourVarAnswers.add("AC");
		}
		for (int i = 0; i < fourVarBottomBoarder.size(); i++) {
			if (fourVarBottomBoarder.get(i) == 0)
				fourVarAnswers.add("ad");
			if (fourVarBottomBoarder.get(i) == 1)
				fourVarAnswers.add("Bd");
			if (fourVarBottomBoarder.get(i) == 2)
				fourVarAnswers.add("Ad");
		}
		for (int i = 0; i < fourVarSideBoarder.size(); i++) {
			if (fourVarSideBoarder.get(i) == 0)
				fourVarAnswers.add("bc");
			if (fourVarSideBoarder.get(i) == 1)
				fourVarAnswers.add("bD");
			if (fourVarSideBoarder.get(i) == 2)
				fourVarAnswers.add("bC");
		}
		if (fourVarBoard[0][0] && fourVarBoard[0][3] && fourVarBoard[3][0]
				&& fourVarBoard[3][3])
			fourVarAnswers.add("bd");
	}
	public static void printThreeVarAnswer() {
		String s = "";
		for (int i = 0; i < threeVarAnswers.size(); i++) {
			s += (threeVarAnswers.get(i) + "+");
		}
		s = s.substring(0, s.length() - 1);
		System.out.print(s);
	}

	public static void printFourVarAnswer() {
		String s = "";
		for (int i = 0; i < fourVarAnswers.size(); i++) {
			s += (fourVarAnswers.get(i) + "+");
		}
		s = s.substring(0, s.length() - 1);
		System.out.print(s);
	}
}
