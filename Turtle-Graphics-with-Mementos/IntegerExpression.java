
public abstract class IntegerExpression{
	/*IntegerExpression is evaluable statement that returns a Integer
	 *Allows other expressions to easily handle a constant or variable
	 */
	public abstract Integer evaluate(Context context);
	public abstract IntegerExpression copy();
	
	
	
	
}
