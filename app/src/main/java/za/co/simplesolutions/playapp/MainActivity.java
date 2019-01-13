package za.co.simplesolutions.playapp;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    private static final String LOG_TAG = MainActivity.class.getSimpleName();
    public static final String EXTRA_MESSAGE = "za.co.simplesolutions.extra.MESSAGE";
    private EditText messageEditText;
    private static final int TEXT_REQUEST = 1;
    private TextView replyHeaderTxt;
    private TextView replyValueText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        messageEditText = findViewById(R.id.editText_main);
        replyHeaderTxt = findViewById(R.id.reply_header_txt);
        replyValueText = findViewById(R.id.reply_txt);
        Log.i(MainActivity.class.getName(), "started...");
    }

    public void launchSecondActivity(View view) {

        Log.d(LOG_TAG, "Button clicked");

        String message = messageEditText.getText().toString();
        Intent intent = new Intent(this, ReplyActivity.class);
        intent.putExtra(EXTRA_MESSAGE, message);
        startActivityForResult(intent, TEXT_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == TEXT_REQUEST) {
            if(resultCode == RESULT_OK) {
                String reply = data.getStringExtra(ReplyActivity.EXTRA_REPLY);
                replyHeaderTxt.setVisibility(View.VISIBLE);
                replyValueText.setText(reply);
                replyValueText.setVisibility(View.VISIBLE);
            }
        }
    }
}
