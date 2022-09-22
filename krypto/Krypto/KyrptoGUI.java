// Andrew Sheinberg

import java.awt.EventQueue;
import javax.swing.JFrame;
import javax.swing.JButton;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import javax.swing.JTextField;
import javax.swing.JLabel;
import java.awt.Font;

public class KyrptoGUI {

	private JFrame frame;
	private JTextField target;
	private JTextField Card2;
	private JTextField Card3;
	private JTextField Card1;
	private JTextField Card4;
	private JTextField Card5;
	private JTextField answer;

	public static void main(String[] args) {
		EventQueue.invokeLater(new Runnable() {
			public void run() {
				try {
					KyrptoGUI window = new KyrptoGUI();
					window.frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	public KyrptoGUI() {
		initialize();
	}
	private void initialize() {
		frame = new JFrame();
		frame.setBounds(100, 100, 450, 300);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(null);
		
		JButton solveButton = new JButton("Solve");
		solveButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {
				int one, two, three, four, five, t;
				one = Integer.parseInt(Card1.getText());
				two = Integer.parseInt(Card2.getText());
				three = Integer.parseInt(Card3.getText());
				four = Integer.parseInt(Card4.getText());
				five = Integer.parseInt(Card5.getText());
				t = Integer.parseInt(target.getText());
				ArrayList<Integer> numberOrder = Krypto_Solver.findOrder(one,two,three,four,five,t);
				int i1=0, i2=0, i3=0, i4=0, i5=0;
				i1=numberOrder.get(0);
				i2=numberOrder.get(1);
				i3=numberOrder.get(2);
				i4=numberOrder.get(3);
				i5=numberOrder.get(4);
				ArrayList<Integer> operationOrder = Krypto_Solver.findOperations(i1,i2,i3,i4,i5,t);
				ArrayList<String> stringOperations;
				stringOperations = new ArrayList<String>();
				String [] operations = {"+", "-", "*", "/"};
				 if (numberOrder.size()>0 && operationOrder.size()>0){
					for (int i=0;i<4;i++){
						stringOperations.add(operations[operationOrder.get(i)]);
					}
					answer.setText((numberOrder.get(0) + stringOperations.get(0) + numberOrder.get(1) + stringOperations.get(1) + numberOrder.get(2) + stringOperations.get(2) + numberOrder.get(3) + stringOperations.get(3) + numberOrder.get(4)));
				}
				else
					answer.setText("No solution is possible");
			}
		});
		solveButton.setBounds(276, 98, 117, 29);
		frame.getContentPane().add(solveButton);
		
		JLabel lblEnterYourKrypto = new JLabel("Enter Your Five Krypto Numbers Below");
		lblEnterYourKrypto.setBounds(16, 43, 255, 16);
		frame.getContentPane().add(lblEnterYourKrypto);
		
		JLabel lblEnterYourTarget = new JLabel("Enter Your Target Krypto Number");
		lblEnterYourTarget.setBounds(37, 111, 214, 16);
		frame.getContentPane().add(lblEnterYourTarget);
		
		target = new JTextField();
		target.setBounds(75, 139, 134, 28);
		frame.getContentPane().add(target);
		target.setColumns(10);
		
		Card2 = new JTextField();
		Card2.setBounds(79, 71, 31, 28);
		frame.getContentPane().add(Card2);
		Card2.setColumns(10);
		
		Card3 = new JTextField();
		Card3.setColumns(10);
		Card3.setBounds(122, 71, 31, 28);
		frame.getContentPane().add(Card3);
		
		Card1 = new JTextField();
		Card1.setColumns(10);
		Card1.setBounds(37, 71, 31, 28);
		frame.getContentPane().add(Card1);
		
		Card4 = new JTextField();
		Card4.setColumns(10);
		Card4.setBounds(166, 71, 31, 28);
		frame.getContentPane().add(Card4);
		
		Card5 = new JTextField();
		Card5.setColumns(10);
		Card5.setBounds(209, 71, 31, 28);
		frame.getContentPane().add(Card5);
		
		answer = new JTextField();
		answer.setBounds(153, 220, 134, 28);
		frame.getContentPane().add(answer);
		answer.setColumns(10);
		
		JLabel lblTheCombinationIs = new JLabel("The Combination Is:");
		lblTheCombinationIs.setBounds(153, 192, 134, 16);
		frame.getContentPane().add(lblTheCombinationIs);
		
		JLabel lblCreatedByAndrew = new JLabel("Created By Andrew Sheinberg");
		lblCreatedByAndrew.setBounds(237, 256, 207, 16);
		frame.getContentPane().add(lblCreatedByAndrew);
		
		JLabel lblKyptoSolver = new JLabel("Kypto Solver");
		lblKyptoSolver.setFont(new Font("Lucida Grande", Font.BOLD, 14));
		lblKyptoSolver.setBounds(177, 15, 94, 16);
		frame.getContentPane().add(lblKyptoSolver);
	}
}