import java.util.*;
public class Program {
	//Program takes in a queue of statements or a queue,context and turtle
	Queue<AbstractExpression> statements;
	private Turtle turtle;
	Context context;
	public Program(Queue<AbstractExpression> statements) {
		//Default for New Program
		this(statements,new Context(),new Turtle());
	}

	public Program(Queue<AbstractExpression> statements,Context context, Turtle turtle) {
		//Used when running repeat blocks
		this.statements = statements;
		this.turtle = turtle;
		this.context = context;
	}
	public Turtle getTurtle() {
		return turtle;
	}
	public void runProgram() {
		//loops through all abstract expressions and evaluates them
		for(AbstractExpression abstExpression:statements)
			abstExpression.evaluate(context,turtle);
	}
}
