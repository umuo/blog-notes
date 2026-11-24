package example.refund;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 启动只处理合成退款数据的 Spring Boot 演练服务。
 */
@SpringBootApplication
public class RefundApplication {
    public static void main(String[] args) {
        SpringApplication.run(RefundApplication.class, args);
    }
}
