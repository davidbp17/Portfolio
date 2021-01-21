public class TurnExpression extends AbstractExpression{
	//Expression for turning the turtle
	private IntegerExpression expr;
	private Memento state;
	public TurnExpression(IntegerExpression expr) {
		this.expr = expr;
	}
	public Memento getMemento() {
		return state;
	}
	public void evaluate(Context context,Turtle turtle) {
		//turns turtle and saves state
		turtle.turn(expr.evaluate(context));
		state = turtle.createMemento();
	}
	public void accept(Visitor visitor) {
		visitor.visit(this);
	}
	public TurnExpression copy() {
		return new TurnExpression(expr);
	}
}
