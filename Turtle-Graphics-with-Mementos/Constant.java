
public class Constant extends IntegerExpression {
	//Wrapper class for Integer that is an IntegerExpression
	private Integer integer;
	public Constant(Integer integer) {
		this.integer = integer;
	}
	public Integer evaluate(Context context) {
		return integer;
	}
	public IntegerExpression copy() {
		return new Constant(integer);
	}

}
