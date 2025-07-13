from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
model = GoogleGenerativeAI(model='gemini-1.5-flash')

classification_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một trợ lý xuất sắc'),
    ('human', 'từ {feedback}, hãy phân loại giúp tôi thành một trong các từ sau (chỉ trả về một từ): '
    'tích cực, tiêu cực, trung lập, cần tư vấn thêm')
])

classification_chain = classification_template | model | StrOutputParser()

positive_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một trợ lý xuất sắc'),
    ('human', 'tạo một lời cảm ơn chân thành với phản hồi tích cực: {feedback}')
])

negative_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một trợ lý xuất sắc'), 
    ('human', 'tạo một lời xin lỗi chân thành về lời phàn nàn {feedback} này')
])

neutual_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một trợ lý xuất sắc'),
    ('human', 'Tạo yêu cầu để biết thêm chi tiết cho phản hồi trung lập {feedback} này')
])

escalate_template = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là một trợ lý xuất sắc'),
    ('human', 'Tạo tin nhắn để chuyển phản hồi {feedback} này đến một tác nhân con người.')
])

branches = RunnableBranch(
    (lambda x: x.strip().lower() == 'tích cực', positive_template | model | StrOutputParser()),
    (lambda x: x.strip().lower() == 'tiêu cực', negative_template | model | StrOutputParser()),
    (lambda x: x.strip().lower() == 'trung lập', neutual_template | model | StrOutputParser()),
    (escalate_template | model | StrOutputParser()),
)

chain = classification_chain | branches

answer = "Sản phẩm tệ quá. Nó hỏng chỉ sau một lần sử dụng và chất lượng rất kém."
#answer = "Sản phẩm tuyệt vời. Tôi thực sự thích sử dụng nó và thấy nó rất hữu ích."

result = chain.invoke({'feedback': answer})
print(result)