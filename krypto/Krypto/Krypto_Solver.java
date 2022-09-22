import java.util.*;
public class Krypto_Solver {
	private static ArrayList<Integer> numberOrder;
	private static ArrayList<Integer> operationOrder;
	private static ArrayList<String> stringOperations;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		int a = in.nextInt();
		int b = in.nextInt();
		int c = in.nextInt();
		int d = in.nextInt();
		int e = in.nextInt();
		int target = in.nextInt();
		findOrder(a,b,c,d,e,target);
		int i1=0,i2=0,i3=0,i4=0,i5=0;
		if (numberOrder.size()==5){
			i1 = numberOrder.get(0);
			i2 = numberOrder.get(1);
			i3 = numberOrder.get(2);
			i4 = numberOrder.get(3);
			i5 = numberOrder.get(4);
		}
		findOperations(i1,i2,i3,i4,i5,target);
		printAnswer();
	}
	public static ArrayList<Integer> findOrder(int a, int b, int c, int d, int e, int target){
		numberOrder=new ArrayList<Integer>();
		int [] order = {a, b, c, d, e};
		int iOne, iTwo, iThree, iFour, iFive;
		for (int i=0;i<5;i++){
			for (int j=0;j<5;j++){
				for (int k=0;k<5; k++){
					for (int l=0;l<5;l++){
						for (int m=0;m<5;m++){
							if (i!=j && i!=k && i!=l && i!=m && j!=k && j!=l && j!=m && k!=l && k!=m && l!=m){
								iOne = order[i];
								iTwo = order[j];
								iThree = order[k];
								iFour = order[l];
								iFive = order[m];
								if (isCombo(iOne, iTwo, iThree,iFour,iFive,target)){
									numberOrder.add(iOne);
								    numberOrder.add(iTwo);
								    numberOrder.add(iThree);
								    numberOrder.add(iFour);
								    numberOrder.add(iFive);
		
								}
							}
						}
					}
				}
			}
		}
		return numberOrder;
	}
	public static boolean isCombo(int a, int b, int c, int d, int e, int target){
		double first, second, third, fourth;
		for (int i=0;i<4;i++){
			for (int j=0;j<4;j++){
				for (int k=0;k<4;k++){
					for (int l=0;l<4;l++){
						if (i==0){
							first = a + b;
						}
						else if (i==1){
							first = a-b;
						}
						else if (i==2){
							first = a*b;
						}
						else{
							first = (double)a/b;
						}
						if (j==0){
							second = first + c;
						}
						else if (j==1){
							second = first - c;
						}
						else if (j==2){
							second = first * c;
						}
						else{
							second = first/c;
						}
						if (k==0){
							third = second + d;
						}
						else if (k==1){
							third = second - d;
						}
						else if (k==2){
							third = second * d;
						}
						else{
							third = second/d;
						}
						if (l==0){
							fourth = third + e;
						}
						else if (l==1){
							fourth = third - e;
						}
						else if (l==2){
							fourth = third * e;
						}
						else{
							fourth = third/e;
						}
						if (fourth==target){
							return true;
						}
					}
				}
			}
		}
		return false;
	}
	public static ArrayList<Integer> findOperations(int a, int b, int c, int d, int e, int target){
		operationOrder = new ArrayList<Integer>();
		double first, second, third, fourth;
		for (int i=0;i<4;i++){
			for (int j=0;j<4;j++){
				for (int k=0;k<4;k++){
					for (int l=0;l<4;l++){
						if (i==0){
							first = a + b;
						}
						else if (i==1){
							first = a-b;
						}
						else if (i==2){
							first = a*b;
						}
						else{
							first = (double)a/b;
						}
						if (j==0){
							second = first + c;
						}
						else if (j==1){
							second = first - c;
						}
						else if (j==2){
							second = first * c;
						}
						else{
							second = first/c;
						}
						if (k==0){
							third = second + d;
						}
						else if (k==1){
							third = second - d;
						}
						else if (k==2){
							third = second * d;
						}
						else{
							third = second/d;
						}
						if (l==0){
							fourth = third + e;
						}
						else if (l==1){
							fourth = third - e;
						}
						else if (l==2){
							fourth = third * e;
						}
						else{
							fourth = third/e;
						}
						if (fourth==target){
							operationOrder.add(i);
							operationOrder.add(j);
							operationOrder.add(k);
							operationOrder.add(l);
						}
					}
				}
			}
		}
		return operationOrder;
	}
	public static void printAnswer(){
		stringOperations = new ArrayList<String>();
		String [] operations = {"+", "-", "*", "/"};
		if (numberOrder.size()==5 && operationOrder.size()==4){
			for (int i=0;i<4;i++){
				stringOperations.add(operations[operationOrder.get(i)]);
			}
			System.out.println(numberOrder.get(0) + stringOperations.get(0) + numberOrder.get(1) + stringOperations.get(1) + numberOrder.get(2) + stringOperations.get(2) + numberOrder.get(3) + stringOperations.get(3) + numberOrder.get(4));
		}
		else
			System.out.println("No solution is possible");
	}
}