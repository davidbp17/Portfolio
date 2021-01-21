import java.util.*;
public class RepeatExpression extends AbstractExpression{
	/*RepeatExpression repeats a statement a given number of times
	 * Takes in a block of statements and an integer expression
	 * The integer expression represents how many times block will loop
	 */
	private IntegerExpression intExp;
	private Queue<AbstractExpression> originalStatements;
	private Queue<AbstractExpression> statements;
	private Memento state;
	public RepeatExpression(IntegerExpression intExpression, Queue<AbstractExpression> statements) {
		this.intExp = intExpression;
		originalStatements = statements;
		this.statements = new LinkedList<AbstractExpression>();
	}
	public Memento getMemento() {
		return state;
	}
	public void evaluate(Context context,Turtle turtle) {	
		int repNum = intExp.evaluate(context);
		//The block get rolled out into another queue so that the visitor can visit all statements
		for(int i = 0; i < repNum;i++) {
			for(AbstractExpression abstExpression:originalStatements) {
				AbstractExpression clone = abstExpression.copy();
				statements.add(clone);
				clone.evaluate(context, turtle);
			}
		}
		//saves state
		state = turtle.createMemento();
	}
	public void accept(Visitor visitor) {
		//loops through all expressions and accepts visitors
		for(AbstractExpression abstExpression:statements)
			abstExpression.accept(visitor);
		visitor.visit(this);
	}
	public RepeatExpression copy() {
		return new RepeatExpression(intExp,originalStatements);
	}

}
