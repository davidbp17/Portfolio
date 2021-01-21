import java.util.*;

import junit.framework.TestCase;
public class InterpreterTest extends TestCase{
	public InterpreterTest() {
		
	}
	public void testInterpreter() throws Exception{
		//Tests general Interpreter and makes sure no errors are thrown
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Test.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		new Program(parsedStatements).runProgram();
		assert(true);
	}
	public void testTurtle() {
		//Tests turtle moving functions
		Turtle turtle = new Turtle();
		assertTrue(turtle.isPenUp());
		assertEquals(turtle.getLocation().getX(),0.0);
		assertEquals(turtle.getLocation().getY(),0.0);
		turtle.turn(45);
		assertEquals(turtle.direction(),45);
		turtle.move(5);
		assertEquals(turtle.getLocation().getX(),3.5355);
		assertEquals(turtle.getLocation().getY(),3.5355);
		turtle.move(-5);
		assertEquals(turtle.getLocation().getX(),0.0);
		assertEquals(turtle.getLocation().getY(),0.0);
	}
	public void testMove() {
		//Tests move expression
		Queue<AbstractExpression> move = new LinkedList<AbstractExpression>();
		move.add(new MoveExpression(new Constant(10)));
		Program testProgram = new Program(move);
		testProgram.runProgram();
		assertEquals(testProgram.getTurtle().getLocation().getX(),10.0);
	}
	public void testTurn() {
		//Tests turn expression
		Queue<AbstractExpression> turn = new LinkedList<AbstractExpression>();
		turn.add(new TurnExpression(new Constant(90)));
		Program testProgram = new Program(turn);
		testProgram.runProgram();
		assertEquals(testProgram.getTurtle().direction(),90);
	}
	public void testPenDown() {
		//Tests penDown expression
		Queue<AbstractExpression> penDown = new LinkedList<AbstractExpression>();
		penDown.add(new PenDownExpression());
		Program testProgram = new Program(penDown);
		testProgram.runProgram();
		assertFalse(testProgram.getTurtle().isPenUp());
	}
	public void testFileDemo() throws Exception{
		//Tests general Interpreter and makes sure no errors are thrown
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Test.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program testDemo = new Program(parsedStatements);
		testDemo.runProgram();
		Turtle turtle = testDemo.getTurtle();
		assertEquals(turtle.getLocation().getX(),22.9904);
		assertEquals(turtle.getLocation().getY(),27.5);
	}
	public void testSquareDraw()throws Exception {
		//Verifies that running the draw square function ends back at (0.0,0,0)
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Square.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		assertFalse(program.getTurtle().isPenUp());
		assertEquals(program.getTurtle().getLocation().getX(),0.0);
		assertEquals(program.getTurtle().getLocation().getY(),0.0);
	}
	public void testSquareLoopDraw()throws Exception {
		//Tests the LoopSquare program and use of repeat statement
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\LoopSquare.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		assertFalse(program.getTurtle().isPenUp());
		assertEquals(program.getTurtle().getLocation().getX(),0.0);
		assertEquals(program.getTurtle().getLocation().getY(),0.0);
	}
	public void testTurtleVisitor()throws Exception {
		/*Runs the square program and implements the visitor pattern
		 * checks every statement to verify it did the expected command
		 */
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Square.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		TurtleVisitor turtleVisitor = new TurtleVisitor();
		for(AbstractExpression abstExpression: parsedStatements)
			abstExpression.accept(turtleVisitor);
		Iterator<Memento> states = turtleVisitor.getStates().iterator();
		Turtle curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.getLocation().getX(),0.0);
		assertEquals(curTurtle.getLocation().getY(),0.0);
		curTurtle = (Turtle)states.next();
		assertFalse(curTurtle.isPenUp());
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.getLocation().getX(),15.0);
		assertEquals(curTurtle.getLocation().getY(),0.0);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.direction(),90);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.getLocation().getX(),15.0);
		assertEquals(curTurtle.getLocation().getY(),15.0);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.direction(),180);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.getLocation().getX(),0.0);
		assertEquals(curTurtle.getLocation().getY(),15.0);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.direction(),270);
		curTurtle = (Turtle)states.next();
		assertEquals(curTurtle.getLocation().getX(),0.0);
		assertEquals(curTurtle.getLocation().getY(),0.0);
	}
	public void testDistanceVisitor()throws Exception {
		//Tests the distance visitor, should be 60
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\Square.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		DistanceVisitor distVisitor = new DistanceVisitor();
		for(AbstractExpression abstExpression: parsedStatements)
			abstExpression.accept(distVisitor);
		assertEquals(distVisitor.getTotalDistance(),60);
	}
	public void testLoopDistanceVisitor()throws Exception {
		//Tests the distance visitor when it comes to repeat statements
		Parser parser = new Parser("C:\\Users\\David Douglas\\eclipse\\TurtleGraphics\\src\\LoopSquare.txt");
		Queue<AbstractExpression> parsedStatements = parser.parse();
		Program program = new Program(parsedStatements);
		program.runProgram();
		DistanceVisitor distVisitor = new DistanceVisitor();
		for(AbstractExpression abstExpression: parsedStatements)
			abstExpression.accept(distVisitor);
		assertEquals(distVisitor.getTotalDistance(),60);
	}



}
