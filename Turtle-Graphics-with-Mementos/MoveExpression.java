
public class MoveExpression extends AbstractExpression{
	//Moves the turtle a given distance
	IntegerExpression expr;
	//Global distance variable for the DistanceVisitor to retrieve
	Integer distance;
	Memento state;
	//Only requires an IntegerExpression
	public MoveExpression(IntegerExpression expr) {
		this.expr = expr;
	}
	
	public Memento getMemento() {
		return state;
	}
	//getDistance is for DistanceVisitor
	public Integer getDistance() {
		return distance;
	}
	
	public void evaluate(Context context, Turtle turtle) {
		//Turtle moves distance and state is saved
		distance = expr.evaluate(context);
		turtle.move(distance);
		state = turtle.createMemento();
	}
	
	public void accept(Visitor visitor) {
		visitor.visit(this);
	}
	
	public MoveExpression copy() {
		return new MoveExpression(expr);
	}
}
