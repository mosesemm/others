package za.co.simplesolutions.playapp;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

public class ReplyActivity extends AppCompatActivity {

    public static final String EXTRA_REPLY = "za.co.simplesolutions.playapp.extra.REPLY";
    private static final String LOG_TAG = ReplyActivity.class.getSimpleName();
    private EditText messageReply;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.d(LOG_TAG, "----------");
        Log.d(LOG_TAG, "onCreate");

        setContentView(R.layout.activity_reply);

        Intent intent = getIntent();
        String message = intent.getStringExtra(MainActivity.EXTRA_MESSAGE);
        TextView textView = findViewById(R.id.text_message);
        textView.setText(message);

        messageReply = findViewById(R.id.editText_reply);
    }

    @Override
    protected void onStart() {
        super.onStart();


        Log.d(LOG_TAG, "onStart");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.d(LOG_TAG, "onStop");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(LOG_TAG, "onDestroy");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(LOG_TAG, "onPause");
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(LOG_TAG, "onResume");
    }

    public void returnReply(View view) {
        Log.i(LOG_TAG, "return button clicked");

        String reply = messageReply.getText().toString();
        Intent replyIntent = new Intent();
        replyIntent.putExtra(EXTRA_REPLY, reply);
        setResult(RESULT_OK, replyIntent);

        Log.i(LOG_TAG, "End second activity");
        finish();
    }
}
