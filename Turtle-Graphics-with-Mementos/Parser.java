import java.io.*;
import java.util.*;
public class Parser{
	/*Parser takes in a filename and can output a queue of AbstractStatements to be run
	 * Has a global BufferedReader for so repeat blocks can be dealt with recursively
	 */
	BufferedReader programFile;
	public Parser(String filename) throws FileNotFoundException{
		programFile = new BufferedReader(new FileReader(new File(filename)));
	}
	public class LoopNotClosedException extends Exception{
		//Exception class for when loop hasn't been ended
		public LoopNotClosedException(String message) {
			super(message);
		}
	}
	public class InvalidExpressionException extends Exception{
		//Exception class for when loop hasn't been ended
		public InvalidExpressionException(String message) {
			super(message);
		}
	}
	//Initial parse function, not in loop
	public Queue<AbstractExpression> parse() throws IOException, LoopNotClosedException, InvalidExpressionException{
		return parse(false);
	}
	//Parse function takes in a the file and reads every line and converts it to an AbstractExpression
	public Queue<AbstractExpression> parse(boolean inLoop) throws IOException, LoopNotClosedException, InvalidExpressionException {
		Queue<AbstractExpression> statements = new LinkedList<AbstractExpression>();	
		String expression;
		while((expression = programFile.readLine())!=null) {
			if(expression.equals("")) continue;
			//String gets broken up into string array representing the expression
			String[] tokenizedString = tokenizer(expression);
			/*Cases for each possible statement
			 * adds the correct AbstractExpression to the queue
			 */
			if(tokenizedString.length == 1) {		
				if(tokenizedString[0].equals("end")){
					//ends the current loop
					inLoop = false;
					break;
				}
				if(tokenizedString[0].equals("penUp")) {
					statements.add(new PenUpExpression());
					continue;
				}
				if(tokenizedString[0].equals("penDown")) {
					statements.add(new PenDownExpression());
					continue;
				}
			}
			else if(tokenizedString.length == 2) {
				//All lines that are two words long contain an IntegerExpression
				IntegerExpression intExpression = (isVar(tokenizedString[1]))?new VariableExpression(tokenizedString[1]):new Constant(Integer.parseInt(tokenizedString[1]));	;
				if(tokenizedString[0].equals("move")) {
					statements.add(new MoveExpression(intExpression));
					continue;
				}
				if(tokenizedString[0].equals("turn")) {
					statements.add(new TurnExpression(intExpression));
					continue;
				}
				if(tokenizedString[0].equals("repeat")) {
					//the repeat adds a recursive parse that exits on the end statement
					//RepeatExpression take a queue
					statements.add(new RepeatExpression(intExpression,parse(true)));
					continue;
				}
			}
			else {
				//Last possibility is an AssignmentStatement, if its an invalid expression
				if(tokenizedString[1].equals("=")) {
					statements.add(new AssignmentExpression(tokenizedString[0],new Constant(Integer.parseInt(tokenizedString[2]))));
					continue;
				}
			}
			throw new InvalidExpressionException("Invalid Expression");
		}
		//if still in loop throw exception
		if(inLoop) throw new LoopNotClosedException("Loop Not Closed");
		return statements;
	}
	
	public boolean isVar(String terminal) {
		return terminal.startsWith("#");
	}
	public String[] tokenizer(String expression) {
		//removes tab and extra spaces at beginning of statement
		while(expression.charAt(0)==(char)32||expression.charAt(0)==(char)9)
			expression = expression.substring(1);
		String[] tokens;
		//breaks down expression to three strings if contains =
		//removes all extra white space on substrings
		if(expression.contains("=")) {
			tokens = new String[3];
			int breakChar = expression.indexOf("=");
			tokens[0] = expression.substring(0,breakChar).replaceAll(" ", "");
			tokens[1] = "=";
			tokens[2] = expression.substring(breakChar+1).replaceAll(" ", "");
		}
		else if(expression.contains("penUp")||expression.contains("penDown")||expression.contains("end")) {
			//one word expression/command will be a one string array
			tokens = new String[1];
			tokens[0] = expression.replaceAll(" ","");
		}
		else {
			//move/turn/repeat statements, must be separated by a space
			tokens = new String[2];
			int breakChar = expression.indexOf(" ");
			tokens[0] = expression.substring(0,breakChar).replaceAll(" ", "");
			tokens[1] = expression.substring(breakChar+1).replaceAll(" ", "");
		}
		return tokens;
	}
	public static void main(String args[]) throws IOException, LoopNotClosedException, InvalidExpressionException {
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Square.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		TurtleVisitor turtleVisitor = new TurtleVisitor();
		DistanceVisitor distanceVisitor = new DistanceVisitor();
		for(AbstractExpression abstExpression : parsedStatements) {
			abstExpression.accept(turtleVisitor);
			abstExpression.accept(distanceVisitor);		
		}
		Iterator<Memento> mementos = turtleVisitor.getStates().iterator();
		while(mementos.hasNext()) {
			System.out.println(((Turtle)mementos.next()).getLocation());
		}
		System.out.println("Total Distance: "+distanceVisitor.getTotalDistance());
	}
}
