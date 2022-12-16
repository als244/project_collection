package ACSLSeniorYear;
import java.util.*;
public class GREEDY {
	private static int[][] graph;
	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		String input = in.nextLine();
		graph = initialize(input);
		for (int i = 0; i < 10; i++) {
			String s = in.nextLine();
			dijkstra(s);
		}
	}
	public static int[][] initialize(String s) {
		String[] edges = s.split(",");
		s = s.replaceAll(",", "");
		char[] input = s.toCharArray();
		int max = 0;
		for (char x : input) {
			if ((int) (x) > max)
				max = (int) (x);
		}
		graph = new int[max-64][max-64];
		int v1 = 0, v2 = 0;
		for (int i = 0; i < edges.length; i++) {
			v1 = (int)(edges[i].charAt(0))-65;
			v2 = (int)(edges[i].charAt(1))-65;
			graph[v1][v2] = (int) (edges[i].charAt(2)-48);
			graph[v2][v1] = (int) (edges[i].charAt(2)-48);
		}
		return graph;
	}
	public static int minDistance(int dist[], boolean visited[]) {
		int min = Integer.MAX_VALUE;
		int min_index = 0;
		for (int i = 0; i < graph.length; i++) {
			if (!visited[i] && dist[i] <= min) {
				min = dist[i];
				min_index = i;
			}
		}
		return min_index;
	}
	public static void dijkstra(String s) {
		int initial = (int)(s.charAt(0))-65;
		int goal = (int) (s.charAt(1))-65;
		int[] dist = new int[graph.length];
		boolean[] visited = new boolean[graph.length];
		for (int i = 0; i < graph.length; i++) {
			dist[i] = Integer.MAX_VALUE;
			visited[i] = false;
		}
		dist[initial] = 0;
		for (int count = 0; count < graph.length - 1; count++) {
			int currentVertex = minDistance(dist, visited);
			visited[currentVertex] = true;
			for (int i = 0; i < graph.length; i++) {
				if (!visited[i] && graph[currentVertex][i] != 0 
						&& dist[currentVertex] != Integer.MAX_VALUE
						&& dist[currentVertex] + graph[currentVertex][i] < dist[i])
					dist[i] = dist[currentVertex] + graph[currentVertex][i];
			}
		}
		System.out.println(dist[goal]);
	}
}