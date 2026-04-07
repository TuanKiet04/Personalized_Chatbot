import streamlit as st
import json

# ---------- Load personas ----------

@st.cache_data
def load_personas():
    with open("personas.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["cluster_id"]: p for p in data}


# ---------- Mock data ----------

BEHAVIOR_PROMPT = (
    '''Bạn là trợ lý tin tức được tinh chỉnh theo hành vi đọc của người dùng.
    Phân tích lịch sử đọc cho thấy người dùng quan tâm đến: AI và xu hướng công nghệ mới, bảo mật tài khoản và an ninh mạng, các vụ gian lận tài chính và cho vay nặng lãi,
    thị trường chứng khoán và rủi ro đầu tư, pháp luật liên quan đến kinh tế và tội phạm.
    Khi trả lời, hãy ưu tiên góc độ cảnh báo rủi ro thực tiễn, phân tích kỹ thuật khi liên quan đến AI/công nghệ,
    và giải thích pháp luật một cách dễ hiểu.
    '''
)

DEFAULT_SYSTEM_PROMPT = """VAI TRÒ:
Bạn là một trợ lý tóm tắt tin tức AI chuyên nghiệp và trung lập.

NHIỆM VỤ:
Trả lời câu hỏi của người dùng MỘT CÁCH NGẮN GỌN VÀ CHÍNH XÁC, chỉ sử dụng các thông tin được cung cấp trong phần [NGỮ CẢNH].

NGUYÊN TẮC BẮT BUỘC:
1. KHÔNG tự bịa đặt thông tin (No Hallucination).
2. KHÔNG sử dụng kiến thức nền tảng có sẵn của bạn.
3. Nếu [NGỮ CẢNH] không chứa đủ thông tin để trả lời, hãy nói: "Dựa trên các bài báo hiện tại, không có đủ thông tin để trả lời câu hỏi này."
4. Trình bày khách quan, không đưa ra quan điểm cá nhân."""

MOCK_QA = {
    "quy trình bầu chủ tịch nước và thủ tướng khác nhau ở điểm nào?": {
        "baseline": (
            "Điểm khác nhau chính nằm ở khâu đề cử. Chủ tịch nước do Ủy ban Thường vụ Quốc hội trình danh sách đề cử, "
            "trong khi Thủ tướng do Chủ tịch nước trình danh sách để Quốc hội bầu. "
            "Các bước còn lại như thảo luận, bỏ phiếu kín, kiểm phiếu, thông qua nghị quyết và tuyên thệ là tương tự nhau."
        ),
        0: (
            "Về mặt pháp lý, sự khác biệt cốt lõi nằm ở chủ thể đề cử. "
            "Hiến pháp trao quyền trình nhân sự Chủ tịch nước cho Ủy ban Thường vụ Quốc hội, "
            "còn Thủ tướng thì do chính Chủ tịch nước vừa được bầu đề xuất. "
            "Điều này tạo ra chuỗi ủy quyền có tính hệ thống: Quốc hội bầu Chủ tịch nước, "
            "Chủ tịch nước đề cử Thủ tướng, Quốc hội phê chuẩn. "
            "Cả hai chức danh đều cần quá bán tổng số đại biểu để trúng cử, đảm bảo tính chính danh."
        ),
        1: (
            "Hãy hình dung như một màn chuyền gậy tiếp sức. Quốc hội bầu Chủ tịch nước trước — "
            "đây là bước mà Ủy ban Thường vụ Quốc hội đứng ra giới thiệu người. "
            "Xong rồi, chính Chủ tịch nước vừa được bầu lại đứng ra giới thiệu Thủ tướng cho Quốc hội bầu tiếp. "
            "Một người vừa nhận quyền lực, ngay lập tức dùng quyền đó để đề cử người kế tiếp — "
            "cấu trúc này vừa tạo sự liên kết, vừa đảm bảo trách nhiệm giải trình rõ ràng."
        ),
        "behavior": (
            "Về mặt thực chất, đây là câu hỏi về phân quyền đề cử. "
            "Chủ tịch nước do Ủy ban Thường vụ Quốc hội đề cử — tức cơ quan lập pháp tự quyết. "
            "Thủ tướng thì do Chủ tịch nước đề cử — tức người đứng đầu nhà nước chịu trách nhiệm chọn người đứng đầu hành pháp. "
            "Về quy trình sau đó hoàn toàn giống nhau: bỏ phiếu kín, quá bán thì trúng, tuyên thệ nhậm chức. "
            "Cần lưu ý rằng nhân sự cả hai đều đã được Ban chấp hành Trung ương thống nhất từ trước, "
            "nên quy trình tại Quốc hội mang tính chính thức hóa nhiều hơn."
        ),
    },
    "chip ai của huawei so với nvidia h100 có khoảng cách bao xa?": {
        "baseline": (
            "Theo nghiên cứu của DeepSeek, chip Ascend 910C của Huawei đạt công suất bằng khoảng 60% so với H100 của Nvidia. "
            "Huawei đang phát triển chip thế hệ mới Ascend 950PR với kế hoạch xuất xưởng khoảng 750.000 chiếc trong năm nay."
        ),
        0: (
            "Khoảng cách hiện tại là đáng kể nhưng đang thu hẹp nhanh. "
            "Ascend 910C của Huawei chỉ đạt 60% hiệu năng H100 theo DeepSeek — con số này phản ánh rủi ro thực tế "
            "khi các hệ thống AI được xây dựng hoàn toàn trên chip nội địa. "
            "Tuy nhiên, Thâm Quyến đã vận hành cụm 10.000 chip Ascend 910C đạt 11.000 petaflop, "
            "và kế hoạch nâng lên 80.000 petaflop trong năm nay cho thấy Trung Quốc đang bù khoảng cách bằng quy mô. "
            "Với Nvidia đang dừng H200 cho thị trường Trung Quốc, áp lực cạnh tranh địa chính trị "
            "có thể đẩy nhanh chu kỳ đầu tư vào chip nội địa hơn nữa."
        ),
        1: (
            "60% — đó là con số DeepSeek đưa ra khi so sánh Ascend 910C với H100 của Nvidia. "
            "Nghe có vẻ thua xa, nhưng hãy nhìn vào bức tranh lớn hơn: "
            "Huawei đang xuất xưởng 812.000 chip AI trong năm nay, gấp nhiều lần so với vài năm trước. "
            "Và khi bạn xếp chồng 10.000 chip 910C lại, cụm máy tính ở Thâm Quyến đạt 11.000 petaflop — "
            "con số đủ để chạy những mô hình AI khổng lồ. "
            "Câu chuyện ở đây không chỉ là con chip đơn lẻ mạnh hay yếu, "
            "mà là liệu Trung Quốc có thể xây dựng hệ sinh thái AI độc lập không cần Nvidia hay không."
        ),
        "behavior": (
            "Khoảng cách kỹ thuật là 40% theo DeepSeek — Ascend 910C đạt 60% hiệu năng H100. "
            "Nhưng con số quan trọng hơn là thị phần: Nvidia vẫn giữ 55% thị trường chip AI Trung Quốc "
            "dù bị hạn chế xuất khẩu, trong khi Huawei chiếm 50% trong số chip nội địa. "
            "Rủi ro đáng chú ý: Nvidia đã dừng kế hoạch H200 cho Trung Quốc, "
            "còn Trung Quốc yêu cầu các công ty công nghệ ưu tiên chip nội địa. "
            "Đây là cuộc chiến không chỉ về kỹ thuật mà còn về chuỗi cung ứng và địa chính trị — "
            "bất kỳ leo thang nào trong quan hệ Mỹ-Trung đều có thể tác động trực tiếp đến thị trường bán dẫn toàn cầu."
        ),
    },
    "xe điện ảnh hưởng thế nào đến petrolimex trong vài năm tới?": {
        "baseline": (
            "Petrolimex dự báo xe điện bắt đầu tác động rõ rệt từ năm nay nhưng chưa chiếm lĩnh thị trường. "
            "Giai đoạn 2028-2030 sản lượng bán xăng dầu có thể đạt đỉnh hoặc đi ngang. "
            "Doanh nghiệp đặt mục tiêu lợi nhuận sau thuế năm nay giảm 7% xuống 3.380 tỷ đồng, mức thấp nhất 4 năm."
        ),
        0: (
            "Petrolimex đang đối mặt với rủi ro cấu trúc dài hạn, không phải rủi ro chu kỳ. "
            "Ngắn hạn, doanh thu mục tiêu tăng 2% lên 315.000 tỷ nhưng lợi nhuận giảm 7% — "
            "biên lợi nhuận đang bị nén. Trung hạn 2028-2030 là điểm bước ngoặt khi sản lượng có thể đạt đỉnh. "
            "Rủi ro kép: vừa có xe điện thay thế nhu cầu, vừa có chính sách hạn chế xe xăng nội đô từ năm sau. "
            "Cổ phiếu PLX hiện ở 39.000 đồng, tăng 10% từ đầu năm — "
            "thị trường có vẻ chưa định giá đầy đủ rủi ro dài hạn này."
        ),
        1: (
            "Hãy hình dung Petrolimex như một con tàu lớn đang thấy băng trôi từ xa. "
            "Hiện tại, 5.500 cửa hàng xăng dầu của họ vẫn đang hoạt động tốt — "
            "xe điện mới chỉ là cơn gió nhẹ. "
            "Nhưng từ 2028-2030, khi giá xe điện rẻ hơn, trạm sạc dày đặc hơn, "
            "và thành phố cấm xe xăng vào nội đô — đó mới là lúc con tàu cảm nhận rõ sức kéo của dòng chảy. "
            "Câu hỏi thú vị là: liệu Petrolimex có kịp chuyển đổi sang kinh doanh trạm sạc điện "
            "trước khi làn sóng đó ập đến không?"
        ),
        "behavior": (
            "Petrolimex đang trong giai đoạn chuyển tiếp rủi ro cao. "
            "Năm nay lợi nhuận dự kiến giảm 7% — mức thấp nhất 4 năm — dù doanh thu vẫn tăng, "
            "cho thấy chi phí đang tăng nhanh hơn doanh thu. "
            "Hai rủi ro cần theo dõi: thứ nhất là chính sách hạn chế xe xăng nội đô từ 2026 "
            "sẽ ảnh hưởng trực tiếp đến 5.500 cửa hàng ở các thành phố lớn; "
            "thứ hai là xung đột Trung Đông đẩy giá dầu DO tăng, ăn vào biên lợi nhuận. "
            "Với cổ phiếu PLX ở 39.000 đồng và tăng 10% từ đầu năm, "
            "nhà đầu tư nên cân nhắc kỹ rủi ro dài hạn trước khi tham gia ở vùng giá này."
        ),
    },
}

QUESTIONS = list(MOCK_QA.keys())


# ---------- UI ----------

st.set_page_config(page_title="RAG Persona Demo", layout="wide")
st.title("RAG Persona Demo")

personas = load_personas()

with st.sidebar:
    st.markdown("### Cài đặt")
    selected_id = st.radio(
        "Chọn Persona",
        options=list(personas.keys()),
        format_func=lambda x: personas[x]["persona"]["name"],
    )
    st.markdown("---")
    st.markdown("**Mô tả:**")
    st.caption(personas[selected_id]["persona"]["description"])
    st.markdown("---")
    st.markdown("**Câu hỏi gợi ý:**")
    for q in QUESTIONS:
        st.caption(f"• {q}")

persona_system_prompt = personas[selected_id]["persona"]["system_prompt"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Baseline RAG",
    "RAG + Persona",
    "RAG + Behavior",
    "So sánh",
])

def get_answer(question, config):
    q = question.strip().lower()
    if q in MOCK_QA:
        return MOCK_QA[q][config]
    return "Câu hỏi này chưa có trong dữ liệu demo. Vui lòng thử một trong các câu hỏi gợi ý ở sidebar."


# ---------- Tab 1 ----------

with tab1:
    st.subheader("Baseline RAG")
    st.caption("Không có Persona, chỉ retrieve và trả lời.")
    with st.expander("System Prompt"):
        st.code(DEFAULT_SYSTEM_PROMPT, language=None)

    q1 = st.text_input("Câu hỏi", key="q1",
                        placeholder="Thử: " + QUESTIONS[0])
    if st.button("Gửi", key="btn1") and q1:
        st.markdown("**Trả lời:**")
        st.write(get_answer(q1, "baseline"))


# ---------- Tab 2 ----------

with tab2:
    st.subheader("RAG + Persona")
    st.caption(f"Persona: {personas[selected_id]['persona']['name']}")
    with st.expander("System Prompt", expanded=True):
        st.code(persona_system_prompt, language=None)

    q2 = st.text_input("Câu hỏi", key="q2",
                        placeholder="Thử: " + QUESTIONS[1])
    if st.button("Gửi", key="btn2") and q2:
        st.markdown("**Trả lời:**")
        st.write(get_answer(q2, selected_id))


# ---------- Tab 3 ----------

with tab3:
    st.subheader("RAG + Behavior Prompt")
    st.caption("Prompt được sinh ra từ lịch sử đọc của người dùng.")
    with st.expander("Behavior Prompt", expanded=True):
        st.code(BEHAVIOR_PROMPT, language=None)

    q3 = st.text_input("Câu hỏi", key="q3",
                        placeholder="Thử: " + QUESTIONS[2])
    if st.button("Gửi", key="btn3") and q3:
        st.markdown("**Trả lời:**")
        st.write(get_answer(q3, "behavior"))


# ---------- Tab 4 ----------

with tab4:
    st.subheader("So sánh 3 cấu hình")
    st.caption("Cùng 1 câu hỏi, hiển thị kết quả của 3 cấu hình cạnh nhau.")

    configs = [
        ("Baseline",  DEFAULT_SYSTEM_PROMPT),
        ("Persona",   persona_system_prompt),
        ("Behavior",  BEHAVIOR_PROMPT),
    ]

    with st.expander("Xem System Prompts"):
        for label, sp in configs:
            st.markdown(f"**{label}**")
            st.code(sp, language=None)

    q4 = st.text_input("Câu hỏi", key="q4",
                        placeholder="Thử bất kỳ câu hỏi nào ở sidebar")
    if st.button("So sánh", key="btn4") and q4:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Baseline**")
            st.write(get_answer(q4, "baseline"))
        with col2:
            st.markdown(f"**{personas[selected_id]['persona']['name']}**")
            st.write(get_answer(q4, selected_id))
        with col3:
            st.markdown("**Behavior**")
            st.write(get_answer(q4, "behavior"))