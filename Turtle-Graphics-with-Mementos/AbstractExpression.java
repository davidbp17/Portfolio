/*AbstractExpression is the abstract class for all the expressions in Turtle Graphics
 * There is AssignmentExpression, MoveExpression, TurnExpression,
 * PenUpExpression, PenDownExpression and RepeatExpression
 * 
 * Evaluate is the method that is called to move the turtle as the expression commands
 * 
 * Accept accepts a visitor and then sends a copy of the expression to it
 * 
 * Copy is a shallow clone so that if a expression is repeated the program
 * doesn't run the same statement over and over again
 */
public abstract class AbstractExpression{
	public abstract void evaluate(Context context,Turtle turtle);
	public abstract void accept(Visitor visitor);
	public abstract AbstractExpression copy();
}
