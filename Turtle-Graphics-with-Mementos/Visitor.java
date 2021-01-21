
public abstract class Visitor {
	//Visitor class demands visits to all AbstractExpressions
	public abstract void visit(AssignmentExpression assignExpression);
	public abstract void visit(MoveExpression moveExpression);
	public abstract void visit(TurnExpression turnExpression);
	public abstract void visit(PenUpExpression penUpExpression);
	public abstract void visit(PenDownExpression penDownExpression);
	public abstract void visit(RepeatExpression repeatExpression);
}
