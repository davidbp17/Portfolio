
public class PenUpExpression extends AbstractExpression{
	//PenUpExpression changes turtle to penUp state
	private Memento state;
	public PenUpExpression() {
	}
	public void evaluate(Context context,Turtle turtle) {
		//calls turtle.penUp() and saves state
		turtle.penUp();
		state = turtle.createMemento();
	}
	public Memento getMemento() {
		return state;
	}
	public void accept(Visitor visitor) {
		visitor.visit(this);
	}
	public PenUpExpression copy() {
		return new PenUpExpression();
	}
}
