
public class DistanceVisitor extends Visitor {
	//DistanceVisitor just calculates total distance, only really relies on MoveExpression
	private int totalDistance;
	public DistanceVisitor() {
		totalDistance = 0;
	}
	public int getTotalDistance() {
		return totalDistance;
	}
	public void visit(AssignmentExpression assignExpression) {
		
	}
	//adds absolute value of distance moved
	public void visit(MoveExpression moveExpression) {
		totalDistance += Math.abs(moveExpression.getDistance());
	}
	public void visit(TurnExpression turnExpression) {
		
	}
	public void visit(PenUpExpression penUpExpression) {
		
	}
	public void visit(PenDownExpression penDownExpression) {
		
	}
	public void visit(RepeatExpression repeatExpression) {
		
	}

}
